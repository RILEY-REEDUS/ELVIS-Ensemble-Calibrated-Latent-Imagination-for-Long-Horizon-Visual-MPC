import warnings

import gym

from envs.wrappers.tensor import TensorWrapper
from envs.wrappers.SB3Vectorized import SB3Vectorized
from envs.dmcontrol import make_env as make_dm_control_env

warnings.filterwarnings('ignore', category=DeprecationWarning)


def make_env(cfg):
	"""
	Make a vectorized DMControl environment.
	"""
	gym.logger.set_level(40)
	env = make_dm_control_env(cfg)
	if env is None:
		raise ValueError(f'Failed to make environment "{cfg.task}".')
	assert cfg.num_envs == 1 or cfg.get('obs', 'state') == 'rgb', \
		'Vectorized environments only support rgb observations.'
	env = SB3Vectorized(cfg, make_dm_control_env)
	env = TensorWrapper(env)
	try:
		cfg.obs_shape = {k: v.shape for k, v in env.observation_space.spaces.items()}
	except AttributeError:
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape}
	cfg.action_dim = env.action_space.shape[0]
	cfg.episode_length = env.max_episode_steps
	cfg.seed_steps = max(1000, 5 * cfg.episode_length) * cfg.num_envs
	return env
