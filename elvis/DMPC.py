import torch
import torch.nn.functional as F
import math as m

from common import math
from common.scale import RunningScale
from common.DreamWorldModel import DreamWorldModel
from tensordict import TensorDict
from typing import Optional, Union, Dict, Any

import numpy as np
import random

class DMPC(torch.nn.Module):
    """
    Belief-MPC agent: a Dreamer-style RSSM world model paired with TD-MPC2-style
    parallel-MPPI planning over learned latent dynamics.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model = DreamWorldModel(cfg).to(self.device)

        # Mixed precision (bf16 by default; safe on Ampere+ without GradScaler)
        self.use_amp = getattr(cfg, "amp", True)
        self.amp_dtype = getattr(cfg, "amp_dtype", torch.bfloat16)
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=(self.use_amp and self.amp_dtype == torch.float16),
        )

        # Precomputed power buffers used in hot paths (planning / imagined rollouts)
        max_T = max(cfg.horizon + 2, cfg.imag_horizon + 2, cfg.plan_horizon + 2)
        rho = float(cfg.rho)
        gamma_depth = float(getattr(cfg, "gamma_depth", 1.0))
        self.register_buffer(
            "_rho_pows",
            (rho ** torch.arange(max_T, device=self.device)).float(),
            persistent=False,
        )
        self.register_buffer(
            "_depth_pows",
            (gamma_depth ** torch.arange(cfg.plan_horizon, device=self.device)).float(),
            persistent=False,
        )

        # Optimizers: world model, critic ensemble, and policy
        self.wm_optim = torch.optim.Adam([
            {'params': self.model._encoder.image_encoder.parameters()},
            {'params': self.model._encoder.image_decoder.parameters()},
            {'params': self.model._encoder.obs_out_plus_suff_stats.parameters()},
            {'params': self.model._reward.parameters()},
            {'params': self.model._inv_dynamics.parameters()},
            {'params': self.model._encoder.belief_decoder.parameters()},
            {'params': self.model._encoder.belief_dynamics.parameters()},
        ], lr=self.cfg.wm_lr, capturable=False)
        self.v_optim = torch.optim.Adam(
            [{'params': self.model._Qs.parameters()}],
            lr=self.cfg.ac_lr, capturable=False,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(),
            lr=self.cfg.ac_lr, eps=1e-5, capturable=False,
        )
        self.model.eval()

        self.scale = RunningScale(cfg)
        self.Q_variance = RunningScale(cfg)
        self.Q_truncation = RunningScale(cfg)
        self.cfg.iterations += 2 * int(cfg.action_dim >= 20)  # extra MPPI iterations for high-D action spaces
        self.discount = self._get_discount(cfg.episode_length)
        self.trust_horizon = None
        self.plan_horizon = None
        self.register_buffer(
            "_prev_mean_eval",
            torch.zeros(1, self.cfg.num_gmms, self.cfg.plan_horizon, self.cfg.action_dim, device=self.device),
        )
        self.register_buffer(
            "_prev_mean",
            torch.zeros(self.cfg.num_envs, self.cfg.num_gmms, self.cfg.plan_horizon, self.cfg.action_dim, device=self.device),
        )

        if cfg.compile:
            print('Compiling update function with torch.compile...')
            self._update = torch.compile(self._update, backend="inductor", dynamic=True, fullgraph=False, mode="reduce-overhead")

        # Flatten optimizer param groups once for fast grad clipping
        self._wm_params = [p for g in self.wm_optim.param_groups for p in g["params"]]
        self._pi_params = [p for g in self.pi_optim.param_groups for p in g["params"]]
        self._q_params  = [p for g in self.v_optim.param_groups  for p in g["params"]]

        # Compile pure forward/loss builders. Backward/step stays outside compile.
        if getattr(self.cfg, "compile", False):
            self._wm_loss_forward = torch.compile(
                self._wm_loss_forward, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
            self._build_imag_rollout = torch.compile(
                self._build_imag_rollout, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
            self._build_shared_returns = torch.compile(
                self._build_shared_returns, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
            self._actor_loss_reinforce = torch.compile(
                self._actor_loss_reinforce, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
            self._actor_loss_dynamics = torch.compile(
                self._actor_loss_dynamics, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
            self._critic_loss = torch.compile(
                self._critic_loss, backend="inductor", dynamic=False,
                fullgraph=True, mode="reduce-overhead"
            )
                # NEW: compile update/apply functions too
            self._apply_wm_update = torch.compile(
                self._apply_wm_update, backend="inductor", dynamic=False,
                fullgraph=False, mode="reduce-overhead"
            )
            self._apply_pi_update = torch.compile(
                self._apply_pi_update, backend="inductor", dynamic=False,
                fullgraph=False, mode="reduce-overhead"
            )
            self._apply_q_update = torch.compile(
                self._apply_q_update, backend="inductor", dynamic=False,
                fullgraph=False, mode="reduce-overhead"
            )

    @property
    def plan(self):
        _plan_val = getattr(self, "_plan_val", None)
        if _plan_val is not None:
            return _plan_val
        if self.cfg.compile:
            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                    plan = torch.compile(self._plan, dynamic=True, mode="max-autotune-no-cudagraphs")
            else:
                plan = torch.compile(self._plan, dynamic=True, mode="max-autotune-no-cudagraphs")
        else:
            plan = self._plan
        self._plan_val = plan
        return self._plan_val
    
    # ============================================================
    # Helper utils: uncompiled apply-update functions
    def _apply_wm_update(self, total_loss):
        """
        Compiled world-model update.
        """
        self.wm_optim.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._wm_params, self.cfg.grad_clip_norm)
        self.wm_optim.step()
        return grad_norm


    def _apply_pi_update(self, pi_loss):
        """
        Compiled actor update.
        Assumes pi_loss should only affect actor params through optimizer stepping.
        Other modules may still receive grads unless their grads are cleared later,
        but they will not be stepped here.
        """
        self.pi_optim.zero_grad(set_to_none=True)
        pi_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._pi_params, self.cfg.grad_clip_norm)
        self.pi_optim.step()
        return grad_norm


    def _apply_q_update(self, value_loss):
        """
        Compiled critic update.
        """
        self.v_optim.zero_grad(set_to_none=True)
        value_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self._q_params, self.cfg.grad_clip_norm)
        self.v_optim.step()
        self.model.soft_update_target_Q()
        return grad_norm
    
    # ============================================================
    # 1) COMPILED WM FORWARD / LOSS BUILDER
    # Pure tensor code only. No backward/step/update here.
    def _wm_loss_forward(self, obs, action, reward, belief_init, z_t, task=None):
        """
        Returns:
            total_loss
            reward_loss
            kl_z_losses
            vae_losses
            action_losses
            new_beliefs
            next_stoch
            v_real_det
            v_real_tru_det
        """
        offset = 1

        new_beliefs = torch.empty_like(belief_init)
        next_stoch = torch.empty_like(z_t)
        new_beliefs[:offset+1].copy_(belief_init[:offset+1])
        next_stoch[:offset+1].copy_(z_t[:offset+1])

        action_losses = torch.zeros((), device=self.device)
        kl_z_losses   = torch.zeros((), device=self.device)
        vae_losses    = torch.zeros((), device=self.device)

        b_t_minus_1 = belief_init[offset]
        z_t_minus_1 = z_t[offset]

        T1 = obs.shape[0]
        for t in range(offset, T1 - 1):
            next_s = self.model.encode(
                b_t_minus_1, z_t_minus_1, action[t], obs[t], task, train=True
            )
            b_t_minus_1 = next_s["deter"]
            z_t_minus_1 = next_s["stoch"]

            new_beliefs[t + 1].copy_(next_s["deter"])
            next_stoch[t + 1].copy_(next_s["stoch"])

            w = self._rho_pows[t]
            kl_z_losses   = kl_z_losses   + next_s["kl_z"]    * w
            vae_losses    = vae_losses    + next_s["vae_loss"]* w
            action_losses = action_losses + next_s["a_loss"]  * w

        next_ss = {"deter": new_beliefs[offset+1:], "stoch": next_stoch[offset+1:]}
        reward_preds = self.model.reward(next_ss, task)
        reward_tgt = reward[offset+1:]

        T, B, D = reward_preds.shape
        TB = T * B
        preds_flat = reward_preds.view(TB, D)
        targets_flat = reward_tgt.view(TB, *reward_tgt.shape[2:])
        loss_flat = math.soft_ce(preds_flat, targets_flat, self.cfg).squeeze(-1)
        losses = loss_flat.view(T, B)
        discounts = self._rho_pows[:T].view(T, 1)
        reward_loss = (losses * discounts).mean()

        # For running uncertainty stats, return detached tensors to update outside compile
        with torch.no_grad():
            q_preds = self.model.V(next_ss, task, return_type='all')
            V = math.two_hot_inv(q_preds, self.cfg)
            mu  = V.mean(dim=0)
            std = V.std(dim=0, unbiased=False)
            v_real_tru = 1 + std / (mu.abs() + 1e-6)
            v_real = self.cfg.unc_sigma * std + mu

        kl_z_losses = kl_z_losses / self.cfg.horizon
        vae_losses = vae_losses / self.cfg.horizon
        action_losses = action_losses / self.cfg.horizon

        total_loss = (
            self.cfg.reward_coef * reward_loss +
            self.cfg.kl_z_coef * kl_z_losses +
            self.cfg.vae_coef * vae_losses +
            self.cfg.action_coef * action_losses
        )

        return (
            total_loss,
            reward_loss,
            kl_z_losses,
            vae_losses,
            action_losses,
            new_beliefs,
            next_stoch,
            v_real.detach(),
            v_real_tru.detach(),
        )
    
    # ============================================================
    # 2) COMPILED SHARED IMAGINED ROLLOUT
    def _build_imag_rollout(self, new_beliefs, next_stoch, task=None):
        """
        Shared rollout for actor and critic.

        Returns:
            imag_beliefs  [H+1, M, belief_dim]
            imag_stoch    [H+1, M, stoch_dim]
            imag_actions  [H,   M, action_dim]
            imag_entropy  [H,   M] or compatible
        """
        offset = 1
        stoch_dim = self.cfg.discrete_dim * self.cfg.stoch_dim if self.cfg.categorical else self.cfg.stoch_dim

        h0 = new_beliefs[offset:].permute(1, 0, 2).reshape(-1, self.cfg.belief_dim).detach()
        z0 = next_stoch[offset:].permute(1, 0, 2).reshape(-1, stoch_dim).detach()

        # Optional AC start subsampling
        n_pool = h0.shape[0]
        ac_batch_size = min(getattr(self.cfg, "ac_batch_size", n_pool), n_pool)
        if ac_batch_size < n_pool:
            idx = torch.randperm(n_pool, device=self.device)[:ac_batch_size]
            h0 = h0.index_select(0, idx)
            z0 = z0.index_select(0, idx)

        s = {"deter": h0, "stoch": z0}

        deter_list = [s["deter"]]
        stoch_list = [s["stoch"]]
        action_list = []
        entropy_list = []

        for t in range(self.cfg.imag_horizon):
            mean_t, a_t, ent_t, dist_t = self.model.pi_dream_compile(s, task)
            action_list.append(a_t)
            entropy_list.append(ent_t)

            s = self.model.next(s, a_t, task)
            deter_list.append(s["deter"])
            stoch_list.append(s["stoch"])

        imag_beliefs = torch.stack(deter_list, dim=0)
        imag_stoch   = torch.stack(stoch_list, dim=0)
        imag_actions = torch.stack(action_list, dim=0)
        imag_entropy = torch.stack(entropy_list, dim=0)

        return imag_beliefs, imag_stoch, imag_actions, imag_entropy
    
    # ============================================================
    # 3) COMPILED SHARED RETURN / TARGET BUILDER
    # Computes ONE shared graph-connected GAE return.
    def _build_shared_returns(
        self,
        imag_beliefs,
        imag_stoch,
        qvar_offset,
        qvar_value,
        qtrunc_offset,
        qtrunc_value,
        task=None,
        ucb=False,
    ):
        """
        Returns:
            gae_returns          # graph-connected
            baseline_det         # critic baseline detached
            critic_preds         # detached rollout critic preds for critic loss
            slow_imag_values     # detached rollout target values
        """
        H = self.cfg.imag_horizon

        imag_s = {
            "deter": imag_beliefs[:H+1],
            "stoch": imag_stoch[:H+1],
        }
        imag_s_detach = {
            "deter": imag_beliefs[:H+1].detach(),
            "stoch": imag_stoch[:H+1].detach(),
        }

        imag_rewards = self.model.reward(
            {
                "deter": imag_beliefs[:H],
                "stoch": imag_stoch[:H],
            },
            task,
        )
        imag_rewards_scaler = math.two_hot_inv(imag_rewards, self.cfg)

        with torch.no_grad():
            # Detached critic outputs for actor training / uncertainty schedule
            critic_bootstrap_det = self.model.V(imag_s_detach, task, return_type='avg')
            baseline_det = critic_bootstrap_det[:-1].detach()
            critic_preds = self.model.V(imag_s_detach, task, return_type='all')
            critic_vs = math.two_hot_inv(critic_preds, self.cfg)
            critic_mu = critic_vs.mean(dim=0)
            critic_std = critic_vs.std(dim=0, unbiased=False)

            if not ucb:
                critic_real = 1 + critic_std / (critic_mu.abs() + 1e-6)
                norm_v = (critic_real - qtrunc_offset) / (qtrunc_value + 1e-6)
            else:
                critic_real = self.cfg.unc_sigma * critic_std + critic_mu
                norm_v = (critic_real - qvar_offset) / (qvar_value + 1e-6)

            alpha = torch.sigmoid((norm_v - self.cfg.lam_center) / 0.1)
            lam_eff = torch.clamp(self.cfg.gae_lambda * (1.0 - alpha), min=self.cfg.lam_min)

        # Graph-connected bootstrap for shared actor return
        gae_bootstrap = self.model.V(imag_s, task, return_type='avg', detach=False)

        gae_returns = torch.empty_like(imag_rewards_scaler)
        next_return = gae_bootstrap[-1]
        for t in reversed(range(imag_rewards_scaler.shape[0])):
            next_return = imag_rewards_scaler[t] + self.cfg.rho * (
                (1 - lam_eff[t]) * gae_bootstrap[t + 1] + lam_eff[t] * next_return
            )
            gae_returns[t] = next_return
        return gae_returns, baseline_det
    
    # ============================================================
    # 4a) COMPILED ACTOR LOSS: REINFORCE
    def _actor_loss_reinforce(
        self,
        imag_beliefs,
        imag_stoch,
        imag_actions,
        gae_returns,
        baseline_det,
        den,
        task=None,
    ):
        H = self.cfg.imag_horizon

        adv_reinforce = (gae_returns.detach() - baseline_det) / den

        mean_pi, _, entropy_pi, dist_pi = self.model.pi_dream_compile(
            {
                "deter": imag_beliefs[:H].detach(),
                "stoch": imag_stoch[:H].detach(),
            },
            task,
        )
        imag_std = dist_pi[1]
        imag_log_pis = self.gaussian_logprob(
            imag_actions[:H],
            mean_pi[:H],
            imag_std[:H],
        )

        rho = self._rho_pows[:adv_reinforce.shape[0]]
        pi_loss = (
            (
                -self.cfg.entropy_coef * entropy_pi
                - imag_log_pis * adv_reinforce.detach()
            ).mean(dim=(1, 2)) * rho
        ).mean()
        return pi_loss
    
    # ============================================================
    # 4b) COMPILED ACTOR LOSS: DYNAMICS ESTIMATOR
    # Reuses shared rollout entropy directly.
    def _actor_loss_dynamics(
        self,
        imag_entropy,
        gae_returns,
        baseline_det,
        den,
    ):
        adv_dyn = (gae_returns - baseline_det) / den
        rho = self._rho_pows[:adv_dyn.shape[0]]

        if imag_entropy.ndim == adv_dyn.ndim - 1:
            entropy_term = imag_entropy.unsqueeze(-1)
        else:
            entropy_term = imag_entropy

        pi_loss = (
            (
                -(adv_dyn + self.cfg.entropy_coef * entropy_term).mean(dim=(1, 2))
            ) * rho
        ).mean()
        return pi_loss
    
    # ============================================================
    # 5) COMPILED CRITIC LOSS BUILDER
    def _critic_loss(self, gae_returns, imag_beliefs, imag_stoch, task=None):
        H = self.cfg.imag_horizon
        imag_s_detach = {
            "deter": imag_beliefs[:H+1].detach(),
            "stoch": imag_stoch[:H+1].detach(),
        }
        slow_imag_values = self.model.V(imag_s_detach, task, target=True)
        critic_preds = self.model.V(imag_s_detach, task, return_type='all')

        Q, T, B, D = critic_preds[:, :-1, :, :].shape
        TBQ = T * B * Q

        preds_flat = critic_preds[:, :-1, :, :].permute(1, 2, 0, 3).contiguous().view(TBQ, D)
        gae_flat = gae_returns.detach().expand(T, B, Q).contiguous().view(TBQ, 1)
        slow_flat = slow_imag_values[:-1].detach().expand(T, B, Q).contiguous().view(TBQ, 1)

        loss1 = math.soft_ce(preds_flat, gae_flat, self.cfg).squeeze(-1)
        loss2 = math.soft_ce(preds_flat, slow_flat, self.cfg).squeeze(-1)
        loss_flat = loss1 + loss2

        losses = loss_flat.view(T, B, Q)
        discounts = self._rho_pows[:T].view(T, 1, 1)
        value_loss = self.cfg.value_coef * (losses * discounts).mean()
        return value_loss

    def _get_discount(self, episode_length):
        """
        Returns discount factor for a given episode length.
        Simple heuristic that scales discount linearly with episode length.
        Default values should work well for most tasks, but can be changed as needed.

        Args:
            episode_length (int): Length of the episode. Assumes episodes are of fixed length.

        Returns:
            float: Discount factor for the task.
        """
        frac = episode_length/self.cfg.discount_denom
        return min(max((frac-1)/(frac), self.cfg.discount_min), self.cfg.discount_max)

    def save(self, fp):
        """
        Save state dict of the agent to filepath.

        Args:
            fp (str): Filepath to save state dict to.
        """
        torch.save({"model": self.model.state_dict()}, fp)

    def load(self, fp):
        """
        Load a saved state dict from filepath (or dictionary) into current agent.

        Args:
            fp (str or dict): Filepath or state dict to load.
        """
        state_dict = fp if isinstance(fp, dict) else torch.load(fp)
        self.model.load_state_dict(state_dict["model"])

    def save_ckpt(self, fp: str, *, step: Optional[int] = None, extra: Optional[Dict[str, Any]] = None):
        ckpt = {
            "model": self.model.state_dict(),

            # DMPC buffers you actually want to restore for MPPI continuity
            "buffers": {
                "_prev_mean_eval": self._prev_mean_eval.detach().cpu(),
                "_prev_mean": self._prev_mean.detach().cpu(),
            },

            "opt": {
                "wm_optim": self.wm_optim.state_dict() if hasattr(self, "wm_optim") else None,
                "v_optim": self.v_optim.state_dict() if hasattr(self, "v_optim") else None,
                "pi_optim": self.pi_optim.state_dict() if hasattr(self, "pi_optim") else None,
            },

            # use your custom RunningScale API
            "running_scales": {
                "scale": self.scale.state_dict() if hasattr(self, "scale") else None,
                "Q_variance": self.Q_variance.state_dict() if hasattr(self, "Q_variance") else None,
                "Q_truncation": self.Q_truncation.state_dict() if hasattr(self, "Q_truncation") else None,
            },

            "step": int(step) if step is not None else None,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
            "extra": extra or {},
        }
        torch.save(ckpt, fp)

    def load_ckpt(
        self,
        fp,
        *,
        strict_model: bool = True,
        map_location: Union[str, torch.device] = "cpu",
        load_optim: bool = True,
        load_running_scales: bool = True,
        restore_rng: bool = False,
        load_mppi_buffers: bool = True,
        device: Optional[torch.device] = None,
    ):
        ckpt = fp if isinstance(fp, dict) else torch.load(fp, map_location=map_location)

        # 1) model weights
        load_info = self.model.load_state_dict(ckpt["model"], strict=strict_model)

        # 2) move after loading (recommended)
        if device is not None:
            self.device = device
            self.model.to(device)
            # keep DMPC buffers on the same device as before; they were created on self.device in __init__

        # 3) restore MPPI buffers (shape-safe)
        if load_mppi_buffers and "buffers" in ckpt:
            b = ckpt["buffers"]

            if "_prev_mean_eval" in b:
                src = b["_prev_mean_eval"].to(self._prev_mean_eval.device)
                if src.shape == self._prev_mean_eval.shape:
                    self._prev_mean_eval.copy_(src)
                # else: silently skip (common if num_envs/plan_horizon changed)

            if "_prev_mean" in b:
                src = b["_prev_mean"].to(self._prev_mean.device)
                if src.shape == self._prev_mean.shape:
                    self._prev_mean.copy_(src)

        # 4) optimizers
        if load_optim and "opt" in ckpt:
            if hasattr(self, "wm_optim") and ckpt["opt"].get("wm_optim") is not None:
                self.wm_optim.load_state_dict(ckpt["opt"]["wm_optim"])
            if hasattr(self, "v_optim") and ckpt["opt"].get("v_optim") is not None:
                self.v_optim.load_state_dict(ckpt["opt"]["v_optim"])
            if hasattr(self, "pi_optim") and ckpt["opt"].get("pi_optim") is not None:
                self.pi_optim.load_state_dict(ckpt["opt"]["pi_optim"])

        # 5) running scales (your custom API)
        if load_running_scales and "running_scales" in ckpt:
            rs = ckpt["running_scales"]
            if hasattr(self, "scale") and rs.get("scale") is not None:
                self.scale.load_state_dict(rs["scale"])
            if hasattr(self, "Q_variance") and rs.get("Q_variance") is not None:
                self.Q_variance.load_state_dict(rs["Q_variance"])
            if hasattr(self, "Q_truncation") and rs.get("Q_truncation") is not None:
                self.Q_truncation.load_state_dict(rs["Q_truncation"])

        # 6) RNG (optional)
        if restore_rng and "rng" in ckpt:
            try:
                torch.set_rng_state(ckpt["rng"]["torch"])
                if torch.cuda.is_available() and ckpt["rng"]["cuda"] is not None:
                    torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
                np.random.set_state(ckpt["rng"]["numpy"])
                random.setstate(ckpt["rng"]["python"])
            except Exception:
                pass

        ckpt["load_info"] = load_info
        return ckpt

    @torch.no_grad()
    def act(self, b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, \
         t0=False, eval_mode=False, explore=False, ucb=False, task=None):
        """
        Select an action by planning in the latent space of the world model.

        Args:
            obs (torch.Tensor): Observation from the environment.
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (int): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
        """
        obs = obs.to(self.device, non_blocking=True)
        if task is not None:
            task = torch.tensor([task], device=self.device)
        if self.cfg.mpc:
            a, s = self.plan(b_t_minus_1, z_t_minus_1, a_t_minus_1, \
                 obs, t0=t0, eval_mode=eval_mode, explore=explore, ucb=ucb, task=task)
        else:
            s = self.model.encode(b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, task, train=False)
            a = self.model.pi_dream_compile(s, task)[int(not eval_mode)][:]
        return a.cpu(), s

    @torch.no_grad()
    def _estimate_value(self, s, actions, task):
        """Estimate value of a trajectory starting at latent state z and executing given actions."""
        G, discount = 0, 1
        for t in range(self.cfg.plan_horizon):
            reward = math.two_hot_inv(self.model.reward(s, actions[:, t], task), self.cfg)
            s = self.model.next(s, actions[:, t], task)
            G = G + discount * reward
            discount_update = self.discount
            discount = discount * discount_update
        return G + discount * self.model.Q(s, self.model.pi(s, task)[1], task, return_type='avg') # TODO: Should we use mean or sampled action?

    @torch.no_grad()
    def _estimate_value_dream(self, s, actions, task):
        """Estimate value of a trajectory starting at latent state s and executing given actions.
        Note: This is implemented with Value function.
        """
        G, discount = 0, 1
        for t in range(self.plan_horizon):
            reward = math.two_hot_inv(self.model.reward(s, task), self.cfg)
            s = self.model.next(s, actions[:, t], task)
            G = G + discount * reward
            discount_update = self.discount
            discount = discount * discount_update
        return G + discount * self.model.V(s, task, return_type='avg')
    
    @torch.no_grad()
    def _estimate_value_dream_with_unc(self, s, actions, task):
        """
        Returns:
        value: (E, N, 1)    — same as before
        v_sum: (E, N, 1)    — depth-weighted Q-ensemble disagreement
        Shapes:
        s: dict with tensors shaped (E, N, state_dim)
        actions: (E, H, N, A)  where H=self.plan_horizon
        """
        device = actions.device
        E, G, H, N, A = actions.shape
        # E, H, N, _ = actions.shape

        # Reshape helper to keep code simple while relying on model broadcasting
        # We keep (E,G,N,*) dims unchanged; model.* is expected to handle leading dims.
        G_acc  = torch.zeros(E, G, N, 1, device=device)
        V_acc  = torch.zeros(E, G, N, 1, device=device)
        disc   = torch.ones(E, G, N, 1, device=device)
        depthw = torch.ones(E, G, N, 1, device=device)

        gamma = self.discount

        for t in range(H):
            # Reward (your existing two-hot inverse)
            r_t = math.two_hot_inv(self.model.reward(s, task), self.cfg)  # (E,N,1)
            G_acc = G_acc + disc * r_t
            # G = G + discount * r_t

            # Disagreement from Q ensemble on (z, a_t)
            a_t = actions[:, :, t, :, :]                                      # (E,G,N,A)
            q_heads = self.model.V(s, task, return_type="all")                # (Q,E,G,N,1)
            q_heads = math.two_hot_inv(q_heads, self.cfg)
            q_mu  = q_heads.mean(dim=0)                                       # (E,G,N,1)
            q_std = q_heads.std (dim=0, unbiased=False)                       # (E,G,N,1)
            v_t   = q_std / (q_mu.abs() + 1e-6)                               # (E,G,N,1)
            V_acc = V_acc + depthw * v_t

            # Step dynamics
            s = self.model.next(s, a_t, task)

            disc  = disc * gamma
            depthw = depthw * self.cfg.gamma_depth  # e.g., 0.95–0.97

        # Terminal bootstrap (avg or low-quantile)
        V_term = self.model.V(s, task, return_type='avg')  # (E,N,1)
        G_acc = G_acc + disc * V_term
        return G_acc, V_acc
    
    @torch.no_grad()
    def _estimate_value_dream_with_unc_lam(self, s, actions, task, use_lambda=False, use_ucb=False):
        """
        Vectorized version:
        1) roll dynamics once to collect z_t and z_{t+1}
        2) batch-call reward(z_t), V_all(z_t), V_avg(z_{t+1})
        3) compute λ-returns & disagreement

        Args:
        s: dict {deter, stoch} with (E, G, N, Dz)
        actions: (E, G, H, N, A)

        Returns:
        value: (E, G, N, 1)
        v_sum: (E, G, N, 1)
        """
        device = actions.device
        E, G, H, N, A = actions.shape
        Dz_det = s['deter'].shape[-1]
        Dz_sto = s['stoch'].shape[-1]

        gamma = self.discount
        rho = gamma
        gamma_depth = self.cfg.gamma_depth

        # ---- 1) Scan dynamics once; store z_t and z_{t+1} ----
        z = s
        z_seq_d = torch.empty(H, E, G, N, Dz_det, device=device)
        z_seq_s = torch.empty(H, E, G, N, Dz_sto, device=device)
        zn_seq_d = torch.empty_like(z_seq_d)
        zn_seq_s = torch.empty_like(z_seq_s)

        for t in range(H):
            z_seq_d[t] = z['deter']
            z_seq_s[t] = z['stoch']
            a_t = actions[:, :, t, :, :]                   # (E,G,N,A)
            z = self.model.next(z, a_t, task)              # (E,G,N,Dz)
            zn_seq_d[t] = z['deter']
            zn_seq_s[t] = z['stoch']

        z_seq  = {'deter': z_seq_d,  'stoch': z_seq_s}     # (H,E,G,N,*)
        zn_seq = {'deter': zn_seq_d, 'stoch': zn_seq_s}

        # ---- 2) Batch reward & critics over all time steps ----
        # If your reward/V accept arbitrary leading dims (last-dim features), this works directly.
        # Otherwise: flatten (H*E*G*N, Dz) → call → reshape back.
        """Note: Lambda-return computation
        TODO: should we use z_1:H to compute lam_0:H-1 instead?
        use z_0, z_1, ..., z_H-2, z_H-1 to compute
        lam_0, lam_1, ..., lam_H-2, lam_H-1;
        use r_0, r_1, ..., r_H-2, r_H-1 and
        z_1, z_2, ..., z_H-1, z_H to compute lambda-return
        
        ret = V(z_H)
        for t from H-1, H-2, ..., 1, 0:
            ret = r_t + rho * (lam_t * ret + (1 - lam_t) * V(z_t+1))
        """
        r_seq  = math.two_hot_inv(self.model.reward(z_seq,  task), self.cfg)   # (H,E,G,N,1)
        Vn_seq = self.model.V(zn_seq, task, return_type='avg')                 # (H,E,G,N,1)

        q_heads = self.model.V(z_seq, task, return_type='all')                 # (Q,H,E,G,N,1)
        q_heads = math.two_hot_inv(q_heads, self.cfg)
        q_mu  = q_heads.mean(dim=0)                                            # (H,E,G,N,1)
        q_std = q_heads.std (dim=0, unbiased=False)                            # (H,E,G,N,1)
        v_seq_tru = 1 + q_std / (q_mu.abs() + 1e-6)                                # (H,E,G,N,1)
        v_seq = self.cfg.unc_sigma * q_std + q_mu                              # UCB

        if use_ucb:
            norm_v = (v_seq - self.Q_variance.offset) / (self.Q_variance.value + 1e-6)
        else:
            # Normalize by std only for truncation
            norm_v = (v_seq_tru - self.Q_truncation.offset) / (self.Q_truncation.value + 1e-6)

        depth_w = self._depth_pows[:H].to(device).view(H, 1, 1, 1, 1)
        v_sum = (depth_w * norm_v).sum(dim=0)  # (E,G,N,1)
        # ---- λ(t) schedule from uncertainty (soft truncation) ----
        if use_lambda:
            lam_base = self.cfg.gae_lambda
            lam_min  = self.cfg.lam_min
            lam_center = self.cfg.lam_center
            lam_k      = 10.0
            alpha = torch.sigmoid( (norm_v - lam_center) * lam_k )              # (H,E,G,N,1)
            lam_t = torch.clamp(lam_base * (1.0 - alpha), min=lam_min)          # (H,E,G,N,1)
        else:
            lam_t = torch.ones_like(r_seq) * self.cfg.gae_lambda                # TD(0)

        # ---- Backward λ-return over H (cheap; leave as tiny loop) ----
        ret = Vn_seq[-1].clone()                                                # (E,G,N,1)
        for t in reversed(range(H)):
            td_boot = (1.0 - lam_t[t]) * Vn_seq[t] + lam_t[t] * ret
            ret = r_seq[t] + rho * td_boot                                      # (E,G,N,1)

        value = ret                                                             # (E,G,N,1)
        return value, v_sum # truncated lambda-return, ucb bonus

    def _get_plan_cache(self, *, E: int, G: int, H: int, S: int, A: int, P0: int):
        """
        Cache shape-only tensors and reusable scratch buffers for planning.

        Caches are keyed by (E,G,H,S,A,P0). This lets eval/train keep separate buffers
        because E differs (1 vs num_envs).
        """
        key = (E, G, H, S, A, P0)
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = {}
            self._plan_cache = cache

        buf = cache.get(key, None)
        if buf is not None and buf["device"] == self.device:
            return buf

        # ---------- build once ----------
        device = self.device

        # Per-mode P_g schedule (linear P0 -> S-P0)
        if P0 > 0:
            P_g = torch.linspace(float(P0), float(S - P0), steps=G, device=device).round().to(torch.int64)
            P_g.clamp_(0, S)
            P_max = int(P_g.max().item())
            P_min = int(P_g.min().item())
        else:
            P_g = torch.zeros(G, device=device, dtype=torch.int64)
            P_max, P_min = 0, 0

        R_max = S - P_min

        if R_max > 0:
            r = torch.arange(R_max, device=device, dtype=torch.int64)    # (R_max,)
            idx = (P_g.view(G, 1) + r.view(1, R_max)).clamp_(0, S - 1)   # (G, R_max)

            # Small template index; expand to (E,G,H,R_max,A) per call without allocating new storage
            idx_tmpl = idx.view(1, G, 1, R_max, 1)                       # (1,G,1,R,1)

            noise = torch.empty(E, G, H, R_max, A, device=device)
            rand_actions = torch.empty(E, G, H, R_max, A, device=device)
        else:
            idx = None
            idx_tmpl = None
            noise = None
            rand_actions = None

        # Scratch buffers that you reuse each plan() call
        actions = torch.empty(E, G, H, S, A, device=device)
        mean    = torch.empty(E, G, H, A, device=device)
        std     = torch.empty(E, G, H, A, device=device)

        buf = {
            "device": device,
            "P_g": P_g,
            "P_max": P_max,
            "P_min": P_min,
            "R_max": R_max,
            "idx": idx,
            "idx_tmpl": idx_tmpl,       # (1,G,1,R,1)
            "noise": noise,             # (E,G,H,R,A)
            "rand_actions": rand_actions,# (E,G,H,R,A)
            "actions": actions,         # (E,G,H,S,A)
            "mean": mean,               # (E,G,H,A)
            "std": std,                 # (E,G,H,A)
        }
        cache[key] = buf
        return buf

    @torch.no_grad()
    def _plan(self, b_t_minus_1, z_t_minus_1, a_t_minus_1,
            obs, t0=False, eval_mode=False, explore=False, ucb=False, task=None):

        torch.compiler.cudagraph_mark_step_begin()

        s = self.model.encode(b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, task, train=False)
        z0 = {'deter': s['deter'], 'stoch': s['stoch']}

        G = self.cfg.num_gmms
        H = self.cfg.plan_horizon
        S = self.cfg.num_samples
        P0 = self.cfg.num_pi_trajs
        A = self.cfg.action_dim
        E = 1 if eval_mode else self.cfg.num_envs

        self.plan_horizon = H
        self._prev_mean_buffer = self._prev_mean_eval if eval_mode else self._prev_mean

        # ---- cache everything shape-only & scratch buffers ----
        buf = self._get_plan_cache(E=E, G=G, H=H, S=S, A=A, P0=P0)
        P_g   = buf["P_g"]
        P_max = buf["P_max"]
        R_max = buf["R_max"]

        actions = buf["actions"]
        mean    = buf["mean"]
        std     = buf["std"]

        # mask per call (task may vary)
        mask = None

        # ---- init mean/std without allocating ----
        mean.zero_()
        std.fill_(self.cfg.max_std)
        if not t0:
            mean[:, :, :-1].copy_(self._prev_mean_buffer[:, :, 1:])

        # ---- (1) pi trajectories: cannot be cached, but allocate at most P_max and write into actions ----
        if P_max > 0:
            pi_actions = torch.empty(E, G, H, P_max, A, device=self.device)
            _z = {k: v.unsqueeze(1).unsqueeze(2).repeat(1, G, P_max, 1) for k, v in z0.items()}
            for t in range(H - 1):
                pi_actions[:, :, t] = self.model.pi_dream_compile(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, :, t], task)
            pi_actions[:, :, -1] = self.model.pi_dream_compile(_z, task)[1]
            if mask is not None:
                pi_actions.mul_(mask)
            actions[:, :, :, :P_max].copy_(pi_actions)

        # ---- expand z for rollouts (depends on current s; cannot cache) ----
        z = {k: v.unsqueeze(1).unsqueeze(2).repeat(1, G, S, 1) for k, v in z0.items()}

        # ---- pre-expand scatter index view without allocating storage ----
        if R_max > 0:
            idx_scatter = buf["idx_tmpl"].expand(E, G, H, R_max, A)  # view only

        for _ in range(self.cfg.iterations):

            if R_max > 0:
                noise = buf["noise"]
                rand_actions = buf["rand_actions"]

                noise.normal_()
                rand_actions.copy_(torch.addcmul(mean.unsqueeze(3), std.unsqueeze(3), noise))
                rand_actions.clamp_(-1, 1)
                if mask is not None:
                    rand_actions.mul_(mask)

                # scatter random into per-mode slots [P_g .. S-1]
                actions.scatter_(dim=3, index=idx_scatter, src=rand_actions)

            value, v_sum = self._estimate_value_dream_with_unc_lam(
                z, actions, task, use_lambda=True, use_ucb=ucb
            )
            if explore:
                value = value + self.cfg.unc_beta * v_sum

            elite_idxs = torch.topk(value.squeeze(-1), self.cfg.num_elites, dim=2).indices
            elite_value = torch.gather(value, 2, elite_idxs.unsqueeze(-1))
            elite_actions = torch.gather(
                actions, 3,
                elite_idxs.unsqueeze(2).unsqueeze(-1).expand(-1, -1, H, -1, A)
            )

            max_value = elite_value.max(dim=2, keepdim=True).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / (score.sum(dim=2, keepdim=True) + 1e-9)

            mean.copy_((score.unsqueeze(2) * elite_actions).sum(dim=3) / (score.sum(dim=2, keepdim=True) + 1e-9))
            std.copy_(((score.unsqueeze(2) * (elite_actions - mean.unsqueeze(3))**2).sum(dim=3)
                    / (score.sum(dim=2, keepdim=True) + 1e-9)).sqrt())
            std.clamp_(self.cfg.min_std, self.cfg.max_std)

            if mask is not None:
                mean.mul_(mask)
                std.mul_(mask)

        # selection (unchanged)
        E2, G2, H2, K, A2 = elite_actions.shape
        logits = elite_value.squeeze(-1)
        logits = logits - logits.amax(dim=(1, 2), keepdim=True)
        p_flat = logits.exp().reshape(E2, G2 * K)
        p_flat = p_flat / (p_flat.sum(dim=1, keepdim=True) + 1e-9)

        flat_idx = math.gumbel_softmax_sample(p_flat, dim=1)
        g_idx = flat_idx // K
        k_idx = flat_idx % K

        t0_elite_actions = elite_actions[:, :, 0]
        t0_std = std[:, :, 0]
        batch = torch.arange(E2, device=std.device)
        a = t0_elite_actions[batch, g_idx, k_idx, :]
        std0 = t0_std[batch, g_idx, :]

        if not eval_mode:
            a = a + std0 * torch.randn(self.cfg.action_dim, device=std.device)

        self._prev_mean_buffer.copy_(mean)
        a = a.clamp(-1, 1).view_as(a_t_minus_1)
        return a, s
    
    @torch.no_grad()
    def _plan_for_viz(self, b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, num_gmms, num_total_trajs, \
                num_pi_trajs, num_elites, t0=False, eval_mode=False, explore=False, ucb=False, task=None):
        """
        Plan a sequence of actions using the learned world model.

        Args:
            b, z, a (torch.Tensor): Latent state from which to plan.
            obs: Current observation from the environment.
            num_gmms: Number of Gaussians in the GMM.
            num_total_trajs: Number of total trajectories per mode. NOTE: total trajs should be same to ensure fairness
            num_pi_trajs: Number of prior trajectories per mode. NOTE: prior trajs should be same to ensure fairness
            num_elites: Number of elite trajectories per mode. NOTE: elite trajs should be same to ensure fairness
            t0 (bool): Whether this is the first observation in the episode.
            eval_mode (bool): Whether to use the mean of the action distribution.
            task (Torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: Action to take in the environment.
            TensorDict: Agent internal memory.
        """
        torch.compiler.cudagraph_mark_step_begin()
        # Sample policy trajectories
        s = self.model.encode(b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, task, train=False) # batch_size X state_dim
        z = {'deter': s['deter'], 'stoch': s['stoch']}
        self.plan_horizon = self.cfg.plan_horizon
        if eval_mode:
            num_envs = 1
            self.register_buffer("_prev_mean_eval", torch.zeros(1, num_gmms, \
                                self.cfg.plan_horizon, self.cfg.action_dim, device=self.device))
            self._prev_mean_buffer = self._prev_mean_eval
        else:
            num_envs = self.cfg.num_envs
            self._prev_mean_buffer = self._prev_mean
        if num_pi_trajs > 0:
            pi_actions = torch.empty(num_envs, num_gmms, self.plan_horizon, num_pi_trajs, self.cfg.action_dim, device=self.device)
            # Our implementation's z is a dict of deter and stoch parts
            _z = {key: tensor.unsqueeze(1).unsqueeze(2).repeat(1, num_gmms, num_pi_trajs, 1) for key, tensor in z.items()}
            for t in range(self.plan_horizon-1):
                pi_actions[:, :, t] = self.model.pi_dream_compile(_z, task)[1]
                _z = self.model.next(_z, pi_actions[:, :, t], task)
            pi_actions[:, :, -1] = self.model.pi_dream_compile(_z, task)[1]

        # Initialize state and parameters
        z = {key: tensor.unsqueeze(1).unsqueeze(2).repeat(1, num_gmms, num_total_trajs, 1) for key, tensor in z.items()}
        mean = torch.zeros(num_envs, num_gmms, self.plan_horizon, self.cfg.action_dim, device=self.device)
        std = self.cfg.max_std*torch.ones(num_envs, num_gmms, self.plan_horizon, self.cfg.action_dim, device=self.device)
        if not t0:
            mean[:, :, :-1] = 0
        actions = torch.empty(num_envs, num_gmms, self.plan_horizon, num_total_trajs, self.cfg.action_dim, device=self.device)
        if num_pi_trajs > 0:
            actions[:, :, :, :num_pi_trajs] = pi_actions

        # Iterate MPPI
        for _ in range(self.cfg.iterations):
            # Sample actions
            r = torch.randn(num_envs, num_gmms, self.plan_horizon, num_total_trajs-num_pi_trajs, self.cfg.action_dim, device=std.device)
            actions_sample = mean.unsqueeze(3) + std.unsqueeze(3) * r
            actions_sample = actions_sample.clamp(-1, 1)
            actions[:, :, :, num_pi_trajs:] = actions_sample
            value, v_sum = self._estimate_value_dream_with_unc_lam(z, actions, task, use_lambda=True, use_ucb=ucb)  # (E,N,1) each
            # value, v_sum = self._estimate_value_dream_with_unc(z, actions, task)  # (E,N,1) each
            if explore:
                # Normalize disagreement with running stats (from REAL (s,a) only)
                # norm = (v_sum - self.Q_variance.offset) / (self.Q_variance.value + 1e-6)  # (E,N,1)
                value = value + self.cfg.unc_beta * v_sum 

            elite_idxs = torch.topk(value.squeeze(-1), num_elites, dim=2).indices
            # elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]
            
            # Gather elite values: (E,G,K,1)
            elite_value = torch.gather(value, 2, elite_idxs.unsqueeze(-1))
            elite_actions = torch.gather(actions, 3, elite_idxs.unsqueeze(2).unsqueeze(-1).expand(-1, -1, self.plan_horizon, -1, self.cfg.action_dim))

            # Update parameters
            max_value = elite_value.max(dim=2, keepdim=True).values
            score = torch.exp(self.cfg.temperature * (elite_value - max_value))
            score = score / (score.sum(dim=2, keepdim=True) + 1e-9)
            # Update mean/std per mode
            # mean/std: (E,G,H,A)
            mean = (score.unsqueeze(2) * elite_actions).sum(dim=3) / (score.sum(dim=2, keepdim=True) + 1e-9)
            std  = ((score.unsqueeze(2) * (elite_actions - mean.unsqueeze(3))**2).sum(dim=3) / (score.sum(dim=2, keepdim=True) + 1e-9)).sqrt()
            std  = std.clamp(self.cfg.min_std, self.cfg.max_std)

        E, G, H, K, A = elite_actions.shape
        logits  = elite_value.squeeze(-1)                                   # (E,G,K)
        logits  = logits - logits.amax(dim=(1,2), keepdim=True) # stabilize
        p_flat  = logits.exp().reshape(E, G*K)
        p_flat  = p_flat / (p_flat.sum(dim=1, keepdim=True) + 1e-9)
        # Gumbel sampling over all (mode, elite)
        flat_idx = math.gumbel_softmax_sample(p_flat, dim=1)  # (E,)
        g_idx = flat_idx // K
        k_idx = flat_idx %  K
        t0_elite_actions = elite_actions[:, :, 0]
        t0_std = std[:, :, 0]
        std_viz = std
        batch = torch.arange(E, device=std.device)
        a = t0_elite_actions[batch, g_idx, k_idx, :]
        std = t0_std[batch, g_idx, :]

        # Select action
        if not eval_mode:
            a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
        # self._prev_mean_buffer.copy_(mean)
        a = a.clamp(-1, 1).view_as(a_t_minus_1)
        return a, s, {"mean": mean, "std": std_viz, "elite_value": elite_value}

    def update_pi(self, zs, task, seed_train=False):
        """
        Update policy using a sequence of latent states.

        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            float: Loss of the policy update.
        """
        _, pis, log_pis, _ = self.model.pi(zs, task)
        qs = self.model.Q(zs, pis, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * log_pis - qs).mean(dim=(1,2)) * rho).mean()
        pi_grad_norm = 0.
        if not seed_train:
            pi_loss.backward()
            pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
            self.pi_optim.step()
            self.pi_optim.zero_grad(set_to_none=True)

        return pi_loss.detach(), pi_grad_norm
    
    # TODO: test policy parameterization correctness
    def update_pi_dream(self, zs, task):
        """
        Update policy using a sequence of latent states.
        Args:
            zs (torch.Tensor): Sequence of latent states.
            task (torch.Tensor): Task index (only used for multi-task experiments).
        Returns:
            float: Loss of the policy update.
        """
        mean_, a_t_, entropy, dist = self.model.pi_dream_compile(zs, task)
        qs = self.model.Q(zs, a_t_, task, return_type='avg', detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a weighted sum of Q-values
        rho = torch.pow(self.cfg.rho, torch.arange(len(qs), device=self.device))
        pi_loss = ((self.cfg.entropy_coef * entropy.unsqueeze(-1) - qs).mean(dim=(1,2)) * rho).mean()
        pi_grad_norm = 0.
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(self.model._pi.parameters(), self.cfg.grad_clip_norm)
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)
        return pi_loss.detach(), pi_grad_norm
    
    @torch.no_grad()
    def _td_target(self, next_s, reward, task):
        """
        Compute the TD-target from a reward and the observation at the following time step.

        Args:
            next_z (torch.Tensor): Latent state at the following time step.
            reward (torch.Tensor): Reward at the current time step.
            task (torch.Tensor): Task index (only used for multi-task experiments).

        Returns:
            torch.Tensor: TD-target.
        """
        pi = self.model.pi(next_s, task)[0] # TODO: Should we use mean or sampled action?
        discount = self.discount
        return reward + discount * self.model.Q(next_s, pi, task, return_type='min', target=True)

    def _update(self, obs, action, reward, belief, z_t, indices, seed_train=False, ucb=False, task=None):
        train_metrics = self.compute_main_loss(obs, action, reward, belief, z_t, indices, task, \
                                 seed_train=seed_train, ucb=ucb)
        return train_metrics

    def update(self, buffer, seed_train=False, ucb=False):
        """
        Main update function. Corresponds to one iteration of model learning.
        Args:
            buffer (common.buffer.Buffer): Replay buffer.
        Returns:
            dict: Dictionary of training statistics.
        """
        (obs, action, reward, belief, z_t, task), indices = buffer.sample_with_indices()
        kwargs = {}
        if task is not None:
            kwargs["task"] = task
        torch.compiler.cudagraph_mark_step_begin()
            # Wrap the *call* to the compiled function
        if self.use_amp:
            with torch.amp.autocast("cuda", dtype=self.amp_dtype):
                return self._update(obs, action, reward, belief, z_t, indices,
                                    seed_train=seed_train, ucb=ucb, **kwargs)
        return self._update(obs, action, reward, belief, z_t, indices, \
                      seed_train=seed_train, ucb=ucb, **kwargs)
    
    def random_frame_mask(obs: torch.Tensor, prob_mask: float) -> torch.Tensor:
        """
        Randomly sets a fraction p_mask of entire observation frames to zero.
        Args:
            obs (torch.Tensor): Batch of observations, shape (B, C, H, W).
            prob_mask (float): Probability of masking each frame (between 0 and 1).
        Returns:
            torch.Tensor: Tensor of the same shape as obs, where a random subset
                        of frames are zeroed out.
        """
        # obs: (batch_size, channels, height, width)
        batch_size = obs.shape[0]
        mask = torch.rand(batch_size, device=obs.device) < prob_mask  # :contentReference[oaicite:3]{index=3}
        # Create a copy of obs to avoid in-place modification, if desired.
        obs_masked = obs.clone()
        # For frames where mask[i] == True, set entire frame to zero.
        obs_masked[mask] = 0  # zero out all channels, pixels :contentReference[oaicite:4]{index=4}
        return obs_masked
    
    def gaussian_logprob(self, x, mean, std):
        """
        Log‑prob of `x` under an independent Gaussian N(mean, std**2).

        Shapes: (..., D)  →  (..., 1)
        All ops are ATen, so the function is torch.compile‑safe.
        """
        log_std = torch.log(std + 1e-6)  # (..., D)
        eps      = (x - mean) / torch.exp(log_std)          # (..., D)
        residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)  # Σ_i
        log2pi   = m.log(2 * m.pi)
        D        = x.size(-1)
        return residual - 0.5 * log2pi * D
        
    def compute_main_loss(self, obs, action, reward, belief_init, z_t, indices,
                        task=None, seed_train=False, ucb=False):
        """
        New workflow:
        1) compiled WM forward/loss
        2) uncompiled WM update
        3) compiled shared imagined rollout
        4) compiled shared return builder
        5) compiled actor loss
        6) compiled critic loss
        7) uncompiled actor update
        8) uncompiled critic update
        """
        self.wm_optim.zero_grad(set_to_none=True)
        self.pi_optim.zero_grad(set_to_none=True)
        self.v_optim.zero_grad(set_to_none=True)
        self.model.train()

        # ------------------------------------------------------------
        # 1) WORLD MODEL FORWARD / LOSS
        (
            total_loss,
            reward_loss,
            kl_z_losses,
            vae_losses,
            action_losses,
            new_beliefs,
            next_stoch,
            v_real_det,
            v_real_tru_det,
        ) = self._wm_loss_forward(obs, action, reward, belief_init, z_t, task)

        # Running uncertainty stats outside compile
        with torch.no_grad():
            self.Q_variance.update(v_real_det.reshape(-1, 1))
            self.Q_truncation.update(v_real_tru_det.reshape(-1, 1))

        # ------------------------------------------------------------
        # 2) WORLD MODEL UPDATE
        grad_norm = self._apply_wm_update(total_loss)

        # Default tensors for seed_train path
        value_loss = torch.zeros((), device=self.device)
        pi_loss = torch.zeros((), device=self.device)
        pi_grad_norm = torch.zeros((), device=self.device)

        if not seed_train:
            # --------------------------------------------------------
            # 3) SHARED IMAGINED ROLLOUT
            imag_beliefs, imag_stoch, imag_actions, imag_entropy = self._build_imag_rollout(
                new_beliefs, next_stoch, task
            )

            # Convert running-scale scalars to tensors once, outside compile
            qvar_offset = torch.as_tensor(self.Q_variance.offset, device=self.device, dtype=imag_beliefs.dtype)
            qvar_value  = torch.as_tensor(self.Q_variance.value,  device=self.device, dtype=imag_beliefs.dtype)
            qtr_offset  = torch.as_tensor(self.Q_truncation.offset, device=self.device, dtype=imag_beliefs.dtype)
            qtr_value   = torch.as_tensor(self.Q_truncation.value,  device=self.device, dtype=imag_beliefs.dtype)

            # --------------------------------------------------------
            # 4) SHARED GAE RETURN + CRITIC TARGETS
            gae_returns, baseline_det = self._build_shared_returns(
                imag_beliefs,
                imag_stoch,
                qvar_offset,
                qvar_value,
                qtr_offset,
                qtr_value,
                task=task,
                ucb=ucb,
            )

            # Running scale for actor normalization from detached return
            with torch.no_grad():
                self.scale.update(gae_returns[0].detach())

            den = torch.clamp(
                torch.as_tensor(self.scale.value, device=self.device, dtype=gae_returns.dtype),
                min=1.0,
            )

            # --------------------------------------------------------
            # 5) ACTOR LOSS
            if not self.cfg.dynamics:
                pi_loss = self._actor_loss_reinforce(
                    imag_beliefs,
                    imag_stoch,
                    imag_actions,
                    gae_returns,
                    baseline_det,
                    den,
                    task=task,
                )
            else:
                pi_loss = self._actor_loss_dynamics(
                    imag_entropy,
                    gae_returns,
                    baseline_det,
                    den,
                )

            # --------------------------------------------------------
            # 6) ACTOR UPDATE
            # Only grads wrt pi params are computed.
            pi_grad_norm = self._apply_pi_update(pi_loss)

                        # --------------------------------------------------------
            # 7) CRITIC LOSS
            value_loss = self._critic_loss(
                gae_returns,
                imag_beliefs,
                imag_stoch,
                task=task
            )

            # --------------------------------------------------------
            # 8) CRITIC UPDATE
            # Only grads wrt Q params are computed.
            Q_grad_norm = self._apply_q_update(value_loss)

        self.model.eval()

        # Build metrics outside compile
        td = TensorDict({}, batch_size=[])
        td.update({
            "vae_loss": self.cfg.vae_coef * vae_losses.detach(),
            "kl_z_loss": self.cfg.kl_z_coef * kl_z_losses.detach(),
            "reward_loss": self.cfg.reward_coef * reward_loss.detach(),
            "value_loss": self.cfg.value_coef * value_loss.detach(),
            "pi_loss": pi_loss.detach(),
            "total_loss": total_loss.detach(),
            "grad_norm": grad_norm.detach() if torch.is_tensor(grad_norm) else torch.as_tensor(grad_norm),
            "pi_grad_norm": pi_grad_norm.detach() if torch.is_tensor(pi_grad_norm) else torch.as_tensor(pi_grad_norm),
            "pi_scale": torch.as_tensor(self.scale.value, device=self.device),
            "action_head_loss": self.cfg.action_coef * action_losses.detach(),
        })
        td.auto_batch_size_()
        return td.detach().mean().cpu(), new_beliefs.detach(), next_stoch.detach(), indices

