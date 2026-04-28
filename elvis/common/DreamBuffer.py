import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from typing import Any, Union

def _to_cpu_tree(x: Any):
    """Recursively move tensors/TensorDicts to CPU for portable saving."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, TensorDict):
        return x.detach().to("cpu")
    if isinstance(x, dict):
        return {k: _to_cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        t = [_to_cpu_tree(v) for v in x]
        return type(x)(t)
    return x

class DreamBuffer():
    """
    Replay buffer for Dreamerv3 + TDMPC2 training (extended).
    Now supports storing additional keys: "belief", and belief updates.
    Uses CUDA memory if available, and CPU memory otherwise.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._sampler = SliceSampler(
            num_slices=self.cfg.batch_size,
            end_key=None,
            traj_key='episode',
            truncated_key=None,
            strict_length=True,
        )
        self._batch_size = cfg.batch_size * (cfg.horizon+1)
        self._num_eps = 0

    @property
    def capacity(self):
        """Return the capacity of the buffer."""
        return self._capacity

    @property
    def num_eps(self):
        """Return the number of episodes in the buffer."""
        return self._num_eps

    def _reserve_buffer(self, storage):
        """
        Reserve a buffer with the given storage.
        """
        return ReplayBuffer(
            storage=storage,
            sampler=self._sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=self._batch_size,
        )

    def _init(self, tds):
        """Initialize the replay buffer. Use the first episode to estimate storage requirements."""
        print(f'Buffer capacity: {self._capacity:,}')
        if torch.cuda.is_available():
            mem_free, _ = torch.cuda.mem_get_info()
            bytes_per_step = sum([
                (v.numel()*v.element_size() if not isinstance(v, TensorDict)
                 else sum([x.numel()*x.element_size() for x in v.values()]))
                for v in tds.values()
            ]) / len(tds)
            total_bytes = bytes_per_step * self._capacity
            print(f'Storage required: {total_bytes/1e9:.2f} GB')
            # Heuristic: decide whether to use CUDA or CPU memory
            storage_device = 'cuda:0' if 2.5 * total_bytes < mem_free else 'cpu'
            print(f'Using {storage_device.upper()} memory for storage.')
            self._storage_device = torch.device(storage_device)
        else:
            self._storage_device = torch.device('cpu')
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

    def _prepare_batch(self, td):
        """
        Prepare a sampled batch for training (post-processing).
        Expects `td` to be a TensorDict with batch size T x B.
        Now selects extra keys "belief".
        """
        td = td.select("obs", "action", "reward", "task", "belief", "z_t", strict=False)\
               .to(self._device, non_blocking=True)
        obs = td.get('obs').contiguous()
        action = td.get('action').contiguous()
        reward = td.get('reward').unsqueeze(-1).contiguous()
        belief = td.get('belief').contiguous()
        z_t = td.get('z_t').contiguous()    
        task = td.get('task', None)
        if task is not None:
            task = task[0].contiguous()
        return obs, action, reward, belief, z_t, task

    def add(self, td):
        """
        Add an episode to the buffer.
        The TensorDict `td` must include keys: "obs", "action", "reward", "task", "belief".
        """
        # Assign episode indices.
        td['episode'] = torch.ones_like(td['reward'], dtype=torch.int64) * \
                        torch.arange(self._num_eps, self._num_eps + self.cfg.num_envs)
        td = td.permute(1, 0)
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
        for i in range(self.cfg.num_envs):
            self._buffer.extend(td[i])
        self._num_eps += self.cfg.num_envs
        return self._num_eps

    def sample(self):
        """Sample a batch of subsequences from the buffer."""
        td = self._buffer.sample().view(-1, self.cfg.horizon+1).permute(1, 0)
        return self._prepare_batch(td)
    
    def sample_with_indices(self):
        """
        Sample a batch of subsequences and also return the indices.
        The underlying _sample method returns a tuple (data, info), and info contains the "index" field.
        """
        td, info = self._buffer._sample(self._batch_size)
        # Reshape and permute as in your current sample method:
        td = td.view(-1, self.cfg.horizon+1).permute(1, 0)
        batch = self._prepare_batch(td)
        return batch, info["index"]
    
    def update_latents_inplace(self, indices, new_belief, new_z_t_minus_1s):
        """
        Given indices (as returned by sample_with_indices) and new latent values
        (new_belief), update the underlying storage in-place.
        
        This implementation assumes that self._buffer.storage._storage is either a single tensor
        or a dictionary of tensors containing the key "belief".
        The update uses torch.index_copy_ to write new values along dimension 0.
        """
        storage = self._buffer._storage  # LazyTensorStorage instance

        if isinstance(storage[indices], TensorDict):
            if "belief" in storage[indices].keys():
                # new_belief: [num_samples, belief_dim]
                # indices: a 1D tensor of indices (of length num_samples)
                # underlying["belief"].index_copy_(0, indices, new_belief)
                storage[indices]["belief"] = new_belief.permute(1, 0, 2).reshape(-1, self.cfg.belief_dim)
            else:
                raise KeyError("Key 'belief' not found in storage.")
            if "z_t" in storage[indices].keys():
                stoch_dim = self.cfg.stoch_dim if not self.cfg.categorical else self.cfg.discrete_dim * self.cfg.stoch_dim
                storage[indices]["z_t"] = new_z_t_minus_1s.permute(1, 0, 2).reshape(-1, stoch_dim)
            else:
                raise KeyError("Key 'z_t' not found in storage.")
        elif isinstance(storage[indices], torch.Tensor):
            # If the storage is a single tensor, you must decide how the keys are arranged.
            # This is less common for multi-key storage.
            raise NotImplementedError("In-place update for single tensor storage is not implemented.")
        else:
            raise TypeError("Underlying storage type not recognized for in-place update.")

    def save(self, fp: str):
        """
        Save replay buffer so training can continue.

        Writes CPU tensors for portability. Restores on load to the chosen storage device.
        """
        ckpt = {
            "num_eps": self._num_eps,
            "capacity": self._capacity,
            "storage_device": str(getattr(self, "_storage_device", torch.device("cpu"))),
            "has_buffer": hasattr(self, "_buffer"),
        }

        if not ckpt["has_buffer"]:
            torch.save(ckpt, fp)
            return

        # Preferred: TorchRL checkpoint API
        if hasattr(self._buffer, "state_dict") and callable(self._buffer.state_dict):
            rb_state = self._buffer.state_dict()
            ckpt["replaybuffer_state"] = _to_cpu_tree(rb_state)
            ckpt["format"] = "replaybuffer_state_dict"
        else:
            # Fallback: dump full storage content (TensorDict) + length if accessible
            storage = self._buffer._storage  # LazyTensorStorage
            # most torchrl versions keep underlying TensorDict at storage._storage
            td_storage = getattr(storage, "_storage", None)
            if td_storage is None:
                raise RuntimeError("Cannot find LazyTensorStorage._storage; upgrade TorchRL or use state_dict().")

            ckpt["storage_tensordict"] = _to_cpu_tree(td_storage)
            ckpt["format"] = "lazy_storage_dump"

            # also try to persist cursor/len if present
            for name in ("_cursor", "_len", "_is_full", "_full"):
                if hasattr(storage, name):
                    ckpt[name] = getattr(storage, name)
        torch.save(ckpt, fp)

    def load(self, fp: str, *, map_location: Union[str, torch.device] = "cpu"):
        """
        Load replay buffer from checkpoint.

        NOTE: cfg.horizon / cfg.batch_size / cfg.num_envs should match what produced the buffer,
        otherwise SliceSampler semantics will not match.
        """
        ckpt = torch.load(fp, map_location=map_location)
        self._num_eps = int(ckpt.get("num_eps", 0))

        if not ckpt.get("has_buffer", False):
            # nothing was stored yet
            return

        # Recreate empty buffer with same capacity. Choose storage device:
        # Use saved device if available; otherwise redo your heuristic by creating a dummy init later.
        saved_dev = ckpt.get("storage_device", "cpu")
        self._storage_device = torch.device("cpu")
        self._buffer = self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device)
        )

        fmt = ckpt.get("format", None)

        if fmt == "replaybuffer_state_dict":
            if not (hasattr(self._buffer, "load_state_dict") and callable(self._buffer.load_state_dict)):
                raise RuntimeError("ReplayBuffer.load_state_dict not available in this TorchRL version.")
            # Move state tensors to the target storage device
            state = ckpt["replaybuffer_state"]

            def _to_dev_tree(x: Any):
                if isinstance(x, torch.Tensor):
                    return x.to(self._storage_device, non_blocking=True)
                if isinstance(x, TensorDict):
                    return x.to(self._storage_device, non_blocking=True)
                if isinstance(x, dict):
                    return {k: _to_dev_tree(v) for k, v in x.items()}
                if isinstance(x, (list, tuple)):
                    t = [_to_dev_tree(v) for v in x]
                    return type(x)(t)
                return x

            self._buffer.load_state_dict(_to_dev_tree(state))
            return

        if fmt == "lazy_storage_dump":
            storage = self._buffer._storage
            td_storage = ckpt["storage_tensordict"].to(self._storage_device, non_blocking=True)

            if getattr(storage, "_storage", None) is None:
                raise RuntimeError("Cannot restore: LazyTensorStorage has no _storage attribute on this version.")
            storage._storage = td_storage

            # best-effort restore cursor/len flags if present
            for name in ("_cursor", "_len", "_is_full", "_full"):
                if name in ckpt and hasattr(storage, name):
                    setattr(storage, name, ckpt[name])
            return

        raise RuntimeError(f"Unknown replay buffer checkpoint format: {fmt}")
