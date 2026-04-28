import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import math as m

from torch import distributions as torchd
from common import d_layers

'''
DreamBeliefTracker is an under-the-hood implementation that combines dreamerv3's RNN-based state-space model
with tdmpc2's mppi-based world model.
h_t in dreamerv3 is b_t in this code. z_t in dreamerv3 is also z_t in this code.
tdmpc2 treats s_t ~ p(s_t|o_{t-N:t}) and uses s_t as the state representation.
Following dreamerv3, we use [b_t, z_t] as the state representation.
'''

class OneHotDist(torchd.one_hot_categorical.OneHotCategorical):
    def __init__(self, logits=None, probs=None, unimix_ratio=0.01):
        if logits is not None and unimix_ratio > 0.0:
            probs = F.softmax(logits, dim=-1)
            probs = probs * (1.0 - unimix_ratio) + unimix_ratio / probs.shape[-1]
            logits = torch.log(probs)
            super().__init__(logits=logits, probs=None)
        else:
            super().__init__(logits=logits, probs=probs)

    def mode(self):
        _mode = F.one_hot(
            torch.argmax(super().logits, axis=-1), super().logits.shape[-1]
        )
        return _mode.detach() + super().logits - super().logits.detach()

    def sample(self, sample_shape=(), seed=None):
        if seed is not None:
            raise ValueError("need to check")
        with torch.no_grad():
            sample = super().sample(sample_shape)#.detach()
        probs = super().probs
        while len(probs.shape) < len(sample.shape):
            probs = probs[None]
        sample += probs - probs.detach()
        return sample
    
class GRUCell(nn.Module):
    """
    Custom GRUCell implemented with pure PyTorch ops (no fused THNN kernels).
    Signature matches torch.nn.GRUCell: forward(input, hx) -> hx_next.
    Usage example:
    cell = GRUCell(input_size=32, hidden_size=64, norm=True)
    h0 = torch.zeros(batch, 64)
    out = cell(torch.randn(batch, 32), h0)
    """
    def __init__(self, inp_size: int, hidden_size: int, bias: bool = False, 
                 norm: bool = True, act=torch.tanh, update_bias=-1.):
        super().__init__()
        self.layers = nn.Sequential()
        self.layers.add_module(
            "GRU_linear", nn.Linear(inp_size + hidden_size, 3 * hidden_size, bias=False)
        )
        if norm:
            self.layers.add_module("GRU_norm", nn.LayerNorm(3 * hidden_size, eps=1e-03))
        self.act = act
        self.inp_size = inp_size
        self.hidden_size = hidden_size
        self.update_bias = update_bias

    def forward(self, input: torch.Tensor, hx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: Tensor of shape (batch, input_size)
            hx:    Tensor of shape (batch, hidden_size)
        Returns:
            next_hx: Tensor of shape (batch, hidden_size)
        """
        # Concatenate input and previous hidden state
        x = torch.cat([input, hx], dim=-1)
        # Compute all gates
        gates = self.layers(x)
        # Split into reset, candidate, and update gates
        reset_gate, cand_gate, update_gate = gates.chunk(3, dim=-1)
        # Activations
        reset = torch.sigmoid(reset_gate)
        candidate = self.act(reset * cand_gate)
        update = torch.sigmoid(update_gate + self.update_bias)
        # New hidden state
        next_hx = update * candidate + (1.0 - update) * hx
        return next_hx

# TODO: Reshaping could be done somewhere else
# Belief Dynamics: Recurrent belief memorizer/updator b_{t} = GRU(b_{t-1}, z_{t-1}, a_{t-1})
class BeliefDynamics(nn.Module):
    def __init__(self, belief_dim, stoch_dim, action_dim, mlp_dim, hidden_dim, cfg):
        super().__init__()
        if not cfg.categorical:
            self.imagine_in = d_layers.mlp(in_dim=stoch_dim+action_dim, mlp_dims=mlp_dim, out_dim=hidden_dim)
        else:
            self.imagine_in = d_layers.mlp(in_dim=stoch_dim*cfg.discrete_dim+action_dim, mlp_dims=mlp_dim, out_dim=hidden_dim)
        self.rnn_cell = GRUCell(hidden_dim, belief_dim) # nn.GRUCell(hidden_dim, belief_dim)
    
    def _step_2d(self, b, z, a):
        inp = self.imagine_in(torch.cat([z, a], dim=-1))
        return self.rnn_cell(inp, b)
    
    def forward(self, b_t_minus_1, z_t_minus_1, a_t_minus_1):
        # Assume inputs are 2D: (batch, feature_dim)
        return self._step_2d(b_t_minus_1, z_t_minus_1, a_t_minus_1)
    
# Belief Inverse Dynamics: a_{t-1} = MLP(b_{t-1}, z_{t-1}, embed_{t})
# This provides extra constraint to learn z_t when observation is not good, a similar idea to MuDreamer.
class BeliefInverseDynamics(nn.Module):
    def __init__(self, belief_dim, stoch_dim, latent_dim, action_dim, mlp_dim, cfg):
        super().__init__()
        self.cfg = cfg
        if not cfg.categorical:
            self.inv_dyn = nn.Linear(latent_dim+stoch_dim+belief_dim, action_dim*2)
        else:
            self.inv_dyn = nn.Linear(latent_dim+stoch_dim*cfg.discrete_dim+belief_dim, action_dim*2)

    def forward(self, inv_inp):
        mean, std = self.inv_dyn(inv_inp).chunk(2, dim=-1)
        mean = torch.tanh(mean)
        std = (self.cfg.max_std - self.cfg.min_std) * torch.sigmoid(
			std + 2.0) + self.cfg.min_std                # positive, bounded
        with torch.no_grad():
            eps = torch.randn_like(std)                        # ATen RNG, traceable
        raw_action   = mean + std * eps                        # reparameterised
        sampled_action = raw_action * (self.cfg.absmax / torch.clip(torch.abs(raw_action), min=self.cfg.absmax)).detach()
        eval_action = mean * (self.cfg.absmax / torch.clip(torch.abs(mean), min=self.cfg.absmax)).detach()
        std = (self.cfg.max_std - self.cfg.min_std) * torch.sigmoid(
                std + 2.0) + self.cfg.min_std
        # --------------------------------------------------------------------- #
		# 5. Entropy of *pre‑squash* Normal (same as torchd.Normal.entropy())   #
		#    H = 0.5 * log(2πeσ²)  summed over act_dim                          #
		# --------------------------------------------------------------------- #
        entropy = (
			0.5 * (1.0 + m.log(2 * m.pi)) + torch.log(std)
		).sum(dim=-1, keepdim=True)                            # (..., 1)
        return eval_action, sampled_action, entropy, (mean, std)
# Belief Decoder: prior predictable predictor p(z_t | b_t)
class BeliefDecoder(nn.Module):
    def __init__(self, belief_dim, hidden_dim, stoch_dim, cfg):
        super().__init__()
        if not cfg.categorical:
            self.imagine_out_plus_suff_stats = d_layers.mlp(in_dim=belief_dim, mlp_dims=hidden_dim, \
                                                            out_dim=2*stoch_dim, dropout=cfg.dyn_dropout)
        else:
            self.imagine_out_plus_suff_stats = d_layers.mlp(in_dim=belief_dim, mlp_dims=hidden_dim, \
                                                            out_dim=cfg.discrete_dim*stoch_dim, dropout=cfg.dyn_dropout)
        self.cfg = cfg
        self.log_std_min = torch.tensor(self.cfg.dyn_min_log_std)
        self.log_std_dif = torch.tensor(self.cfg.log_std_max - self.cfg.log_std_min)
    
    def forward(self, b_t):
        dist = None
        if self.cfg.categorical:
            x = self.imagine_out_plus_suff_stats(b_t)
        else:
            x = self.imagine_out_plus_suff_stats(b_t)
        return x
        # # When using ensemble
        # return x.mean(0)

# Reparameterization Trick for sampling
def reparameterize(mu, logvar, min_std=0.1):
    std = torch.exp(0.5 * logvar)
    with torch.no_grad():
        eps = torch.randn_like(std)
    std = std + min_std
    return mu + eps * std, torch.log(std)

def kl_loss(mu_q, logvar_q, mu_p, logvar_p, free_bit=1.0):
    # KL-term clipped by free bit
    kl = 0.5 * (logvar_p - logvar_q + (torch.exp(logvar_q) + (mu_q - mu_p)**2) / torch.exp(logvar_p) - 1)
    # kl is assumed to be a tensor of shape [batch_size, latent_dim]
    kl_clipped = torch.clamp(torch.sum(kl, dim=-1), min=free_bit)
    return torch.mean(kl_clipped), torch.sum(kl, dim=-1)

def kl_loss_cate(post_dist, prior_dist, post_dist_detach, prior_dist_detach, free, dyn_scale, rep_scale):

    rep_loss = value = torchd.kl.kl_divergence(
        post_dist, prior_dist_detach
    )
    dyn_loss = torchd.kl.kl_divergence(
        post_dist_detach, prior_dist
    )
    # this is implemented using maximum at the original repo as the gradients are not backpropagated for the out of limits.
    rep_loss = torch.clip(rep_loss, min=free)
    dyn_loss = torch.clip(dyn_loss, min=free)
    loss = dyn_scale * dyn_loss + rep_scale * rep_loss
    return loss, value, dyn_loss, rep_loss

# KL helper – analytic KL with free‑bits and dyn/rep scales               #
def kl_loss_cate_fast(
    post_probs:  torch.Tensor,
    prior_probs: torch.Tensor,
    free: float        = 0.1,
    dyn_scale: float   = 1.0,
    rep_scale: float   = 0.1,
    eps: float         = 1e-8,
):
    """
    Args
    ----
    post_probs  : (..., stoch, disc)  p(z_t | b_t, o_t)
    prior_probs : (..., stoch, disc)  p(z_t | b_t)        (same shape)
    free        : free‑bits threshold
    dyn_scale   : scale for forward KL  KL(post_detached || prior)
    rep_scale   : scale for reverse KL  KL(post || prior_detached)

    Returns
    -------
    loss      : scaled sum  dyn_scale*dyn_loss + rep_scale*rep_loss
    kl_raw    : KL(post || prior)  (no free‑bits, no detach)
    dyn_loss  : free‑clipped forward KL  (gradients disabled)
    rep_loss  : free‑clipped reverse KL  (gradients pass through post)
    """
    kl_rep = (post_probs * (torch.log(post_probs + eps) -
                            torch.log(prior_probs.detach() + eps))).sum((-2, -1))
    kl_dyn = (post_probs.detach() * (torch.log(post_probs.detach() + eps) -
                            torch.log(prior_probs + eps))).sum((-2, -1))

    rep_loss = torch.clamp(kl_rep, min=free)
    dyn_loss = torch.clamp(kl_dyn, min=free)
    loss     = dyn_scale * dyn_loss + rep_scale * rep_loss
    return loss, kl_rep, dyn_loss, rep_loss

# Posterior helper – produces a straight‑through one‑hot latent and probs  #
def categorical_one_hot_mode(
    logits: torch.Tensor,
    stoch_dim: int,
    discrete_dim: int,
    unimix_ratio: float = 0.01,
):
    """
    Args
    ----
    logits : (..., stoch_dim*discrete_dim)  unnormalised logits
    stoch_dim, discrete_dim : factorised categorical sizes
    unimix_ratio : Dreamer‑style uniform mix for exploration

    Returns
    -------
    z_t   : (..., stoch_dim*discrete_dim)  straight‑through one‑hot sample
    probs : (..., stoch_dim, discrete_dim) smoothed categorical probabilities
    """
    # reshape -> (..., stoch, disc)
    logits = logits.view(*logits.shape[:-1], stoch_dim, discrete_dim)

    # unimix smoothing
    probs  = F.softmax(logits, dim=-1)
    probs  = probs * (1.0 - unimix_ratio) + unimix_ratio / discrete_dim

    # straight‑through hard one‑hot mode
    y_soft = probs
    y_hard = F.one_hot(y_soft.argmax(-1), discrete_dim).type_as(y_soft)
    z_t    = (y_hard + (y_soft - y_soft.detach()))          # STE
    z_t    = z_t.reshape(*z_t.shape[:-2], stoch_dim * discrete_dim)
    return z_t, probs

def gumbel_softmax_st(
    logits: torch.Tensor,
    *,
    stoch_dim: int,
    discrete_dim: int,
    temperature: float = 1.0,
    unimix: float = 0.01,
    hard: bool = True,
    noise = None,
):
    """
    Returns
    -------
    z_t   : (B, stoch_dim, discrete_dim)  one‑hot sample (hard if hard=True)
    probs : same shape, soft probs (for KL etc.)
    """
    # 0) reshape last axis -> (stoch, disc)
    logits = logits.view(*logits.shape[:-1], stoch_dim, discrete_dim)

    # 1) unimix smoothing ----------------------------------------------------
    probs  = torch.softmax(logits, -1)
    probs  = probs * (1 - unimix) + unimix / discrete_dim
    logp   = torch.log(probs)                           # needed for Gumbel trick

    # 2) draw Gumbel noise ---------------------------------------------------
    if noise is None:
        g = -torch.empty_like(logp).exponential_().log()        # Gumbel(0,1)
    else:
        g = noise.view_as(logp)

    y_soft = torch.softmax((logp + g) / temperature, -1)        # relaxed sample

    # 3) straight‑through hard one‑hot --------------------------------------
    if hard:
        y_hard = torch.nn.functional.one_hot(
            y_soft.argmax(-1), discrete_dim
        ).type_as(y_soft)
        z_t = y_hard + (y_soft - y_soft.detach())               # STE
        z_t = y_hard + (probs - probs.detach()) # dreamer style
    else:
        z_t = y_soft

    return z_t.view(*logits.shape[:-2], stoch_dim*discrete_dim), probs

def onehot_sample_st(
    logits,
    stoch_dim,
    discrete_dim,
    temperature=None,
    unimix=0.01,
    use_gumbel=False,
    clip_logits=20,
    eps=1e-6,
):
    logits = logits.reshape(*logits.shape[:-1], stoch_dim, discrete_dim)
    out_dtype = logits.dtype
    logits_f = logits.float()

    if clip_logits is not None:
        logits_f = logits_f.clamp(-float(clip_logits), float(clip_logits))

    logits_f = torch.nan_to_num(logits_f, nan=0.0, posinf=0.0, neginf=0.0)

    probs = torch.softmax(logits_f, dim=-1)

    if unimix > 0.0:
        probs = probs * (1.0 - unimix) + unimix / discrete_dim

    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp_min(0.0)

    row_sum = probs.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(probs, 1.0 / discrete_dim)
    bad_mask = (row_sum <= eps).expand_as(probs)
    probs = torch.where(bad_mask, uniform, probs)
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(eps)

    if use_gumbel:
        logp = probs.clamp_min(eps).log()
        g = -torch.empty_like(logp).exponential_().log()
        hard_idx = (logp + g).argmax(dim=-1)
    else:
        hard_idx = torch.multinomial(probs.reshape(-1, discrete_dim), 1).squeeze(-1)
        hard_idx = hard_idx.reshape(probs.shape[:-1])

    y_hard = F.one_hot(hard_idx, discrete_dim).to(probs.dtype)
    z_t = y_hard + (probs - probs.detach())

    if temperature is not None and temperature < 1e4:
        temp = max(float(temperature), eps)
        probs_soft = torch.softmax(logits_f / temp, dim=-1)
    else:
        probs_soft = probs

    z_t_flat = z_t.reshape(*logits.shape[:-2], stoch_dim * discrete_dim)
    return z_t_flat.to(out_dtype), probs_soft.to(out_dtype)
    
class DreamBeliefTracker(nn.Module):
    """
    Variational Belief tracker using KalmanNet for belief state tracking.
    """
    def __init__(self, params):
        super(DreamBeliefTracker, self).__init__()
        # Initialize LKN with appropriate dimensions
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.params = params
        
        # Will try with deeper world model nets, for VAE-related code, refer to 
        # branch tdmpc_rnn_dreamer on 2025/04/13
        # Posterior flow
        # (batch, c, h, w) -> (batch, hidden), note: hidden = 512 = cfg.latent_dim
        self.image_norm = d_layers.PixelPreprocess()
        self.image_encoder = d_layers.conv(in_shape=(3, 64, 64), num_channels=params.num_channels)
        stoch_dim = params.stoch_dim if not params.categorical else params.discrete_dim * params.stoch_dim
        self.image_decoder = d_layers.Decoder(stoch_dim=stoch_dim, deter_dim=params.belief_dim, num_channels=params.num_channels)
        if not self.params.categorical:
            self.obs_out_plus_suff_stats = d_layers.mlp(in_dim=params.latent_dim + params.belief_dim, mlp_dims=params.latent_dim, \
                                        out_dim=2 * params.stoch_dim)
        else:
            self.obs_out_plus_suff_stats = d_layers.mlp(in_dim=params.latent_dim + params.belief_dim, mlp_dims=params.latent_dim, \
                                        out_dim=params.discrete_dim * params.stoch_dim)
        # Prior flow
        self.belief_dynamics = BeliefDynamics(belief_dim=params.belief_dim, stoch_dim=params.stoch_dim, \
                                action_dim=params.action_dim, mlp_dim=params.mlp_dim, hidden_dim=params.latent_dim, cfg=params)
        self.inverse_dynamics = BeliefInverseDynamics(belief_dim=params.belief_dim, latent_dim=params.latent_dim, stoch_dim=params.stoch_dim, \
                                                      action_dim=params.action_dim, mlp_dim=params.mlp_dim, cfg=params)
        self.belief_decoder = BeliefDecoder(belief_dim=params.belief_dim, hidden_dim=params.latent_dim, \
                                            stoch_dim=params.stoch_dim, cfg=params)
        self.log_std_min = torch.tensor(self.params.dyn_min_log_std)
        self.log_std_dif = torch.tensor(self.params.log_std_max - self.params.log_std_min)

    # @torch.compiler.disable
    def forward(self, b_t_minus_1, z_t_minus_1, a_t_minus_1, o_t, train=False):
        """
        Track belief state using LKN.
        Args:
            o_t: Current observation tensor.
            imagine: imagine during trajectory optimization to save computation.
        """
        # TODO: In dreamer, stoch has dim of (stoch_dim, dyn_discrete)
        # Prior flow  
        # Belief update using RNN NOTE: We use b_t = GRU(b_{t-1}, z_{t-1}, a_{t-1})
        b_t = self.belief_dynamics(b_t_minus_1, z_t_minus_1, a_t_minus_1)
        # Use the belief decoder to predict the prior latent observation \hat{z_t} ~ p(z_t | b_t):
        x_prior = self.belief_decoder(b_t)
        z_t_prior, prior_dist = onehot_sample_st(x_prior, stoch_dim=self.params.stoch_dim, discrete_dim=self.params.discrete_dim)

        # Posterior flow
        embed = self.image_encoder(o_t)
        embed_plus_deter = torch.cat([b_t, embed], dim=-1)
        vae_loss, kl_z, action_loss = 0, 0, 0
        if not self.params.categorical:
            mu, log_std = self.obs_out_plus_suff_stats(embed_plus_deter).chunk(2, dim=-1)
            std = torch.exp(log_std)
            std = 2 * torch.sigmoid(std / 2)
            log_std = torch.log(std)
            z_t, log_std = reparameterize(mu, logvar=2*log_std)
            post_dist = {"mu": mu, "log_std": log_std}
            if train:
                kl_z, kl_z_raw = kl_loss(mu_q=prior_dist["mu"], logvar_q=2*prior_dist["log_std"], \
                                    mu_p=post_dist["mu"], logvar_p=2*post_dist["log_std"], free_bit=0.1)
        else:
            x = self.obs_out_plus_suff_stats(embed_plus_deter)
            z_t, post_probs = onehot_sample_st(x, stoch_dim=self.params.stoch_dim, discrete_dim=self.params.discrete_dim)
            # KL (compile‑friendly)
            prior_probs = prior_dist
            if train:
                kl_z_raw, value, dyn_loss, rep_loss = kl_loss_cate_fast(
                    post_probs,
                    prior_probs,
                    free = self.params.free_bit,
                    dyn_scale = 1.0,
                    rep_scale = 0.1,
                )
        if train:
            deter_plus_stoch = torch.cat([b_t, z_t], dim=-1)
            o_rec = self.image_decoder(deter_plus_stoch)
            o_t_norm = self.image_norm(o_t)
            vae_loss = F.mse_loss(o_rec, o_t_norm, reduction='none').sum(dim=(1, 2, 3)).mean()
            inv_deter_plus_embed = torch.cat([b_t_minus_1, z_t_minus_1, embed], dim=-1)
            a_t_minus_1_pred, a_sample, a_ent, a_dist = self.inverse_dynamics(inv_deter_plus_embed)
            # TODO: optimize KL term between two action distributions, need to put action dist in replay buffer as well.
            # This can also help implement the idea from Bootstrapped MPC.
            action_loss = F.mse_loss(a_sample, a_t_minus_1, reduction='mean')
            kl_z = kl_z_raw.mean()
        B = post_probs.shape[0]
        s = {"stoch": z_t, "deter": b_t, "a_t_minus_1": a_t_minus_1, "probs": post_probs.reshape(B, -1), \
             "vae_loss": vae_loss, "kl_z": kl_z, "a_loss": action_loss}
        return s
    
