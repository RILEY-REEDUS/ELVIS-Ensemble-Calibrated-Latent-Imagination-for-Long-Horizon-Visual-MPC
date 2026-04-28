from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer
from pathlib import Path
from typing import Optional, Tuple

def _is_numeric_stem(path: Path) -> Optional[int]:
    """Return step if filename stem is purely digits, else None."""
    s = path.stem
    return int(s) if s.isdigit() else None


def latest_ckpt_pair_in_dir(d: str) -> Tuple[Optional[str], Optional[str], int]:
    """
    Find the latest *numeric* checkpoint step in directory `d` that has:
      - <step>.pt            (agent checkpoint)
      - <step>.replay.pt     (replay buffer checkpoint)

    Returns: (agent_ckpt_path, replay_ckpt_path, step)
      - If none found, returns (None, None, 0)
      - If agent exists but replay missing, it is skipped (to ensure resumable training)
    """
    d = Path(d)
    if not d.is_dir():
        return None, None, 0

    best_step = 0
    best_agent = None
    best_replay = None

    for agent_p in d.glob("*.pt"):
        # ignore replay files in this loop
        if agent_p.name.endswith(".replay.pt"):
            continue

        step = _is_numeric_stem(agent_p)
        if step is None:
            continue

        replay_p = d / f"{step}.replay.pt"
        if not replay_p.is_file():
            continue  # require replay file to resume training

        if step > best_step:
            best_step = step
            best_agent = agent_p
            best_replay = replay_p

    if best_agent is None:
        return None, None, 0

    return best_agent, best_replay, best_step

def maybe_resume(
    agent,
    replay_buffer,
    fp: str,
    device,
    strict_model: bool = True,
):
	"""
	Resume from:
		- a directory: loads latest numeric pair (<step>.pt + <step>.replay.pt)
		- a file path: if it points to <step>.pt, also requires <step>.replay.pt
	Returns: (start_step, agent_ckpt_path, replay_ckpt_path)
		- If not resuming: (0, None, None)
	"""
	p = fp
	if p is None:
		return 0, None, None

	# Case 1: directory -> pick latest pair
	if p.is_dir():
		agent_path, replay_path, step = latest_ckpt_pair_in_dir(str(p))
		if agent_path is None:
			return 0, None, None
		agent.load_ckpt(agent_path,
			strict_model=strict_model,
			load_optim=True,
			load_running_scales=True,
			restore_rng=True,
			load_mppi_buffers=True,
			device=device)
		replay_buffer.load(replay_path, map_location="cpu")
		return step, agent_path, replay_path

	# Case 2: explicit file
	if p.is_file():
		# If user accidentally passes replay file, map back to agent file
		if p.name.endswith(".replay.pt"):
			agent_p = p.with_name(p.name.replace(".replay.pt", ".pt"))
			replay_p = p
		else:
			agent_p = p
			# require matching replay file
			step = _is_numeric_stem(agent_p)
			replay_p = agent_p.parent / f"{step}.replay.pt" if step is not None else agent_p.with_name(agent_p.stem + ".replay.pt")

		if not agent_p.is_file() or not replay_p.is_file():
			return 0, None, None

		step = _is_numeric_stem(agent_p) or 0

		agent.load_ckpt(
			str(agent_p),
			strict_model=strict_model,
			load_optim=True,
			load_running_scales=True,
			restore_rng=True,
			load_mppi_buffers=True,
			device=device,
		)
		replay_buffer.load(str(replay_p), map_location="cpu")
		return step, str(agent_p), str(replay_p)

	return 0, None, None

class DOnlineTrainer(Trainer):
	"""Trainer class for single-task online dreamer + TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()
		self._should_ucb = False

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	# Use single render_env for rendering
	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		# During eval, batch_size = 1
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = self.env.render_env.reset(), False, 0, 0
			b_t_minus_1 = torch.zeros(1, self.cfg.belief_dim, device=self.agent.device, requires_grad=False)
			z_t_minus_1 = torch.zeros(1, self.cfg.stoch_dim, device=self.agent.device, requires_grad=False)
			if self.cfg.categorical:
				z_t_minus_1 = torch.zeros(1, self.cfg.discrete_dim * self.cfg.stoch_dim, device=self.agent.device, requires_grad=False)
			a_t_minus_1 = torch.zeros(1, self.cfg.action_dim, device=self.agent.device, requires_grad=False).view(1, -1)
			if self.cfg.save_video:
				self.logger.video.init(self.env.render_env, enabled=(i==0))
			while not done:
				torch.compiler.cudagraph_mark_step_begin()
				action, s = self.agent.act(b_t_minus_1, z_t_minus_1, a_t_minus_1, \
								obs, t0=t==0, eval_mode=True, explore=False, ucb=self._should_ucb)
				obs, reward, done, info = self.env.render_env.step(action)
				b_t_minus_1 = s["deter"]
				z_t_minus_1 = s["stoch"]
				a_t_minus_1 = action.to(device=self.agent.device)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env.render_env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		if not self._should_ucb:
			self._should_ucb = bool(np.nanmean(ep_rewards) > self.cfg.ucb_auto_ep_return)
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)
	
	# From vectorized_env branch
	def to_td(self, obs, action=None, reward=None, belief=None, z_t=None):
		"""Creates a TensorDict for a new episode."""
		if isinstance(obs, dict):
			obs = TensorDict(obs, batch_size=(), device='cpu')
		else:
			obs = obs.unsqueeze(0).cpu()
		if action is None:
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.tensor(float('nan')).repeat(self.cfg.num_envs)
		if belief is None:
			belief = torch.tensor(float('nan')).repeat(self.cfg.num_envs, self.cfg.belief_dim)
		if z_t is None:
			z_t = torch.tensor(float('nan')).repeat(self.cfg.num_envs, self.cfg.stoch_dim) if not self.cfg.categorical else \
				torch.tensor(float('nan')).repeat(self.cfg.num_envs, self.cfg.discrete_dim * self.cfg.stoch_dim)
		td = TensorDict(dict(
			obs=obs,
			action=action.unsqueeze(0).cpu(),
			reward=reward.unsqueeze(0).cpu(),
			belief=belief.unsqueeze(0).cpu(),
			z_t=z_t.unsqueeze(0).cpu(),
		), batch_size=(1, self.cfg.num_envs,))
		return td

	# From vectorized_env branch
	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, done, eval_next = {}, torch.tensor(True), False
		#TODO: Add code to load agent if path is available
		resume_fp_agent, resume_fp_replay, self.start_step = latest_ckpt_pair_in_dir(self.logger._model_dir)
		_, str_used_agent, str_used_replay = maybe_resume(self.agent, self.buffer, resume_fp_agent, device=self.agent.device)

		# train for "_steps on top of whatever was already done"
		self._step = self.start_step + self._step
		print(f"Resume ckpt: {str_used_agent}")
		print(f"start_step={self.start_step}  target_step={self.cfg.steps}")

		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % self.cfg.eval_freq == 0:
				eval_next = True
			if self._step % self.cfg.save_freq == 0:
				self.logger.save_agent(self.agent, self.buffer, identifier=str(self._step))

			# Reset environment
			if done.any():
				assert done.all(), 'Vectorized environments must reset all environments at once.'
				# During train, batch_size = num_envs
				if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False
				if self._step > self.start_step: # 0:
					tds = torch.cat(self._tds)
					train_metrics.update(
						episode_reward=tds['reward'].nansum(0).mean(),
						episode_success=info['success'].nanmean(),
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')
					self._ep_idx = self.buffer.add(tds)

				obs = self.env.reset()
				b_t_minus_1 = torch.zeros(self.cfg.num_envs, self.cfg.belief_dim, device=self.agent.device, requires_grad=False)
				z_t_minus_1 = torch.zeros(self.cfg.num_envs, self.cfg.stoch_dim, device=self.agent.device, requires_grad=False)
				if self.cfg.categorical:
					z_t_minus_1 = torch.zeros(self.cfg.num_envs, self.cfg.discrete_dim * self.cfg.stoch_dim, device=self.agent.device, requires_grad=False)
				a_t_minus_1 = torch.zeros(self.cfg.num_envs, self.cfg.action_dim, device=self.agent.device, requires_grad=False).view(self.cfg.num_envs, -1)
				self._tds = [self.to_td(obs)]

			# Collect experience
			if self._step >= self.cfg.seed_steps:
				action, s = self.agent.act(b_t_minus_1, z_t_minus_1, a_t_minus_1, \
								obs, t0=len(self._tds)==1, explore=True, ucb=self._should_ucb)
				b_t_minus_1 = s["deter"]
				z_t_minus_1 = s["stoch"]
				a_t_minus_1 = action.to(device=self.agent.device)
			else:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self._tds.append(self.to_td(obs, action, reward, b_t_minus_1, z_t_minus_1))

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = int(self.cfg.seed_steps / self.cfg.steps_per_update * self.cfg.seed_train_percent)
					print('Pretraining agent on seed data...')
					should_update = True
					seed_train = True
				else:
					num_updates = max(1, int(self.cfg.num_envs / self.cfg.steps_per_update))
					should_update = ((self._step - self.cfg.seed_steps) / self.cfg.num_envs % self.cfg.steps_per_update == 0)
					seed_train = False
				if should_update:
					for _ in range(num_updates):
						_train_metrics, new_beliefs, new_z_t, indices = self.agent.update(self.buffer, seed_train, \
																		ucb=self._should_ucb)
						self.buffer.update_latents_inplace(indices, new_beliefs, new_z_t)
					train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs
	
		self.logger.finish(self.agent)
