from copy import deepcopy

import torch
import torch.nn as nn
import math as m

from common import d_layers, math, init, tools
from tensordict.nn import TensorDictParams
from torch import distributions as torchd

from common.DreamBeliefTracker import onehot_sample_st

class DreamWorldModel(nn.Module):
	"""
	This is the belief-based version of TD-MPC2 implicit world model architecture.
	The latent transition is modelled in a dreamerv3-like fashion.
	Note: latent is expressed as [b_t, z_t], where b_{t+1} = GRU(b_t, z_t, a_t), z_t = enc(o_t).
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = d_layers.belief_tracker(cfg)
		self._dynamics = self._encoder.belief_dynamics
		self._inv_dynamics = self._encoder.inverse_dynamics
		self._stoch_pred = self._encoder.belief_decoder
		stoch_feat = cfg.stoch_dim * cfg.discrete_dim if cfg.categorical else cfg.stoch_dim
		head_in = cfg.belief_dim + stoch_feat
		self._reward = d_layers.mlp(head_in, 2*[cfg.mlp_dim], max(cfg.num_bins, 1))
		self._pi = d_layers.mlp(head_in, 2*[cfg.mlp_dim], 2*cfg.action_dim)
		self._Qs = d_layers.Ensemble([
			d_layers.mlp(head_in, 2*[cfg.mlp_dim], max(cfg.num_bins, 1), dropout=cfg.dropout)
			for _ in range(cfg.num_q)
		])
		self.apply(init.weight_init)
		init.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)
		self.init()

	def init(self):
		# Create params
		self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
		self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

		# Create modules
		with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
			self._detach_Qs = deepcopy(self._Qs)
			self._target_Qs = deepcopy(self._Qs)

		# Assign params to modules
		self._detach_Qs.params = self._detach_Qs_params
		self._target_Qs.params = self._target_Qs_params

	def __repr__(self):
		repr = 'Belief TD-MPC2 World Model\n'
		modules = ['Belief tracker', 'Reward', 'Policy prior', 'Q-functions']
		for i, m in enumerate([self._encoder, self._reward, self._pi, self._Qs]):
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		self.init()
		return self

	def train(self, mode=True):
		"""
		Overriding `train` method to keep target Q-networks in eval mode.
		"""
		super().train(mode)
		self._target_Qs.train(False)
		return self

	def soft_update_target_Q(self):
		"""
		Soft-update target Q-networks using Polyak averaging.
		"""
		self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

	def encode(self, b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, task, train):
		"""
		Encodes an observation into its latent representation.
		Note: This serves as the posterior observation.
		"""
		if self.cfg.belief_rl == 1:
			if self.cfg.obs == 'rgb' and obs.ndim == 5: # obs shape = T, B, C, H, W
				#TODO: Is this actually used?
				return torch.stack([self._encoder(o) for o in obs])
			return self._encoder(b_t_minus_1, z_t_minus_1, a_t_minus_1, obs, train)
		if self.cfg.obs == 'rgb' and obs.ndim == 5:
			return torch.stack([self._encoder[self.cfg.obs](o) for o in obs])
		return self._encoder[self.cfg.obs](obs)
	
	def next(self, s, a, task, dropout=True):
		"""
		Predicts the next latent deterministic state h given the current latents and action.
		"""
		# Use MC-Dropout to remove epistemic uncertainty
		if dropout:
			#TODO: Append s and a, get multiple h_deter
			B = s["deter"].shape[0]
			K = self.cfg.num_dropout_passes
			# The above can be slow, if drop the mc on the deter
			h_deter = self._dynamics(s["deter"], s["stoch"], a)
			h_rep = h_deter.unsqueeze(0).expand(K, *h_deter.shape).reshape(B*K, *h_deter.shape[1:])

			#TODO: Use multiple h_deter to get multiple x_imag, average them, sample one z_imag
			x_imag = self.imagine_z(h_rep, task)
			x_imag_rest = x_imag.shape[1:]
			x_imag = x_imag.reshape(K, B, *x_imag_rest).mean(0)
			z_imag, dist = onehot_sample_st(x_imag, stoch_dim=self.cfg.stoch_dim, discrete_dim=self.cfg.discrete_dim)
		else: 
			h_deter = self._dynamics(s["deter"], s["stoch"], a)
			x_imag = self.imagine_z(h_deter, task)
			z_imag, dist = onehot_sample_st(x_imag, stoch_dim=self.cfg.stoch_dim, discrete_dim=self.cfg.discrete_dim)
		next_s = {"stoch": z_imag, "deter": h_deter, "prob": dist}
		return next_s
	
	# 	Args:
	# 		s: dict with keys {"deter","stoch"} for current latent state
	# 		a: action tensor
	# 		task: task embedding input (if multitask)
	# 		dropout (bool): use MC-dropout ensemble if True
	# 		entropy_gate (dict|None): e.g., {"single": lam1, "path": lam2}
	# 			If provided, will return an extra flag `stop` in the output dict.

	# 	Returns:
	# 		z_imag: sampled stochastic latent fed to GRU
	# 	B = s["deter"].shape[0]
	# 	K = self.cfg.num_dropout_passes if dropout else 1

	# 	# Small epsilon for numerical safety
	# 	eps = 1e-6

	# 		# (5) Sample denoised stochastic latent and update deter GRU
	# 		z_imag = torch.randn_like(cond_mu) * cond_var.sqrt() + cond_mu  # [B, Dz]

	# 		# Collect diagnostics
	# 		out.update({
	# 			"fused_mu": fused_mu, "fused_var": fused_var,
	# 			"epi_var": epi_var, "K_gain": K_gain,
	# 			"cond_mu": cond_mu, "cond_var": cond_var,
	# 			"entropy": H,
	# 		})

	# 		dist = None  # (optional) return a distribution if your API expects it

	# 		out.update({
	# 			"fused_logits": fused_logits,
	# 			"epi_var_log": epi_var_log,
	# 			"alea_var_log": alea_var_log,
	# 			"K_gain": K_gain,
	# 			"entropy": H,
	# 		})

	# 	# ----- (C) Deterministic core update with denoised z -----
	# 	h_next = self._dynamics(h_deter, z_imag, a)

	# 	# Return new stochastic latent, (optional) dist, and diagnostics incl. stop flag
	# 	return {"deter": h_next, "stoch": z_imag, "prob": dist, "diag": out}
	
	def imagine_z(self, h, task):
		"""
		Imagine latent obs z from h, i.e., prior obs from current memory.
		"""
		x_imag = self._stoch_pred(h)
		return x_imag
	
	def value_head(self, s, task):
		"""
		Predicts the value of current state, learned via trying to mimic the critic.
		"""
		z = torch.cat([s["deter"], s["stoch"]], dim=-1)
		V = self._value_head(z)
		return V

	def save_belief_tracker_memory(self):
		"""
		Save belief tracker memory during experience collection.
		This function returns a dict of saved memory
		After batch update, the belief tracker can restore it.
		"""
		return self._encoder.save_state()
	
	def restore_belief_tracker_memory(self, saved_state):
		"""
		Restore belief tracker's internal memory and states.
		Args:
			saved_state: Dict of saved states
		"""
		self._encoder.restore_state(saved_state)

	def reward(self, s, task):
		"""
		Predicts instantaneous (single-step) reward.
		"""
		z = torch.cat([s["deter"], s["stoch"]], dim=-1)
		return self._reward(z)
	
	def action_head(self, s_t, s_t_plus_1, task):
		"""
		Predict current action given current state and next state.
		"""
		z_t_plus_1 = torch.cat([s_t_plus_1["deter"], s_t_plus_1["stoch"]], dim=-1)
		o_rec = self._encoder.image_decoder(z_t_plus_1)
		embed = self._encoder.image_encoder(o_rec)
		inv_deter_plus_embed = torch.cat([s_t["deter"], s_t["stoch"], embed], dim=-1)
		a_t_eval, a_t_sample, ent, dist = self._inv_dynamics(inv_deter_plus_embed)
		return a_t_eval, a_t_sample, ent, dist

	def pi(self, s, task):
		"""
		Samples an action from the policy prior.
		The policy prior is a Gaussian distribution with
		mean and (log) std predicted by a neural network.
		"""
		z = torch.cat([s["deter"], s["stoch"]], dim=-1)
		# Gaussian policy prior
		mu, log_std = self._pi(z).chunk(2, dim=-1)
		log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
		eps = torch.randn_like(mu)
		action_dims = None
		log_pi = math.gaussian_logprob(eps, log_std, size=action_dims)
		pi = mu + eps * log_std.exp()
		mu, pi, log_pi = math.squash(mu, pi, log_pi)
		return mu, pi, log_pi, log_std
	
	def pi_dream(self, s, task):
		"""
		Samples from Gaussian policy prior with different squashing and 
		normalization than TD-MPC2's code
		"""
		z = torch.cat([s["deter"], s["stoch"]], dim=-1)
		# Gaussian policy prior
		mean, std = self._pi(z).chunk(2, dim=-1)
		std = (self.cfg.max_std - self.cfg.min_std) * torch.sigmoid(
                std + 2.0) + self.cfg.min_std
		dist = torchd.normal.Normal(torch.tanh(mean), std)
		dist = tools.ContDist(
			torchd.independent.Independent(dist, 1), absmax=self.cfg.absmax
		)
		sampled_action = dist.sample()
		eval_action = dist.mode()
		entropy = dist.entropy()
		return eval_action, sampled_action, entropy, dist
	
	def pi_dream_compile(
		self,
		s: dict,                       # {"deter": …, "stoch": …}
		task=None,
	):
		"""
		Compile‑friendly Gaussian policy prior with tanh squashing.

		Args
		----
		s            : dict with tensors "deter" and "stoch" (…, D)
		task         : optional task id (for multitask embedding)
		noise        : optional ε ~ N(0,1) tensor.  If None, draws with randn_like.
		return_stats : whether to also return (mean, std)  (for logging)

		Returns
		-------
		eval_action      : deterministic policy action  (mode)      (…, act_dim)
		sampled_action   : stochastic action with STE gradients     (…, act_dim)
		entropy          : Normal entropy *before* tanh             (…, 1)
		(mean, std)      : optional, only if return_stats=True
		"""
		z = torch.cat([s["deter"], s["stoch"]], dim=-1)        # (..., F)
		mean, std = self._pi(z).chunk(2, dim=-1)         # (..., act_dim) each
		mean = torch.tanh(mean)
		std = (self.cfg.max_std - self.cfg.min_std) * torch.sigmoid(
			std + 2.0) + self.cfg.min_std                # positive, bounded
		with torch.no_grad():
			eps = torch.randn_like(std)                        # ATen RNG, traceable
		raw_action   = mean + std * eps                        # reparameterised
		sampled_action = raw_action * (self.cfg.absmax / torch.clip(torch.abs(raw_action), min=self.cfg.absmax)).detach()
		eval_action = mean * (self.cfg.absmax / torch.clip(torch.abs(mean), min=self.cfg.absmax)).detach()

		# --------------------------------------------------------------------- #
		# 5. Entropy of *pre‑squash* Normal (same as torchd.Normal.entropy())   #
		#    H = 0.5 * log(2πeσ²)  summed over act_dim                          #
		# --------------------------------------------------------------------- #
		entropy = (
			0.5 * (1.0 + m.log(2 * m.pi)) + torch.log(std)
		).sum(dim=-1, keepdim=True)                            # (..., 1)
		return eval_action, sampled_action, entropy, (mean, std)

	def V(self, s, task, return_type='min', target=False, detach=False):
		"""
		Predict state value.
		`return_type` can be one of [`min`, `avg`, `all`]:
			- `min`: return the minimum of two randomly subsampled values.
			- `avg`: return the average of two randomly subsampled values.
			- `all`: return all values.
		`target` specifies whether to use the target value networks or not.
		"""
		assert return_type in {'min', 'avg', 'all'}

		z = torch.cat([s["deter"], s["stoch"]], dim=-1)
		if target:
			qnet = self._target_Qs
		elif detach:
			qnet = self._detach_Qs
		else:
			qnet = self._Qs
		out = qnet(z)

		if return_type == 'all':
			return out

		vidx = torch.randperm(self.cfg.num_q, device=out.device)[:2]
		V = math.two_hot_inv(out[vidx], self.cfg)
		if return_type == "min":
			return V.min(0).values
		return V.sum(0) / 2
