from copy import deepcopy
import numpy as np
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecEnvWrapper

class SB3Vectorized(VecEnvWrapper):
    """
    SB3 vectorized environment wrapper for TD-MPC2 online training.
    Uses SB3’s vectorized environments so that you can call custom methods,
    such as get_state, on each individual environment during training.
    Note: This is actually slower than Gym's AsynVecEnv, it is only for debugging!
    """
    def __init__(self, cfg, env_fn, vec_env_cls=DummyVecEnv):
        self.cfg = cfg

        def make():
            _cfg = deepcopy(cfg)
            _cfg.num_envs = 1
            _cfg.seed = cfg.seed #+ np.random.randint(1000)
            print("train env seed is: ", _cfg.seed)
            return env_fn(_cfg)
        
        def make_render_env():
            _cfg = deepcopy(cfg)
            _cfg.num_envs = 1
            _cfg.max_episode_steps = 500 # Set to 500 for evaluation as TD-MPC2
            _cfg.seed = cfg.seed
            return env_fn(_cfg)

        print(f"Creating {cfg.num_envs} environments using {vec_env_cls.__name__}...")
        env_fns = [make for _ in range(cfg.num_envs)]
        env = vec_env_cls(env_fns)
        super().__init__(env)
        self.env = env
        
        # Use a single environment instance to retrieve observation and action spaces.
        single_env = make()
        self.observation_space = single_env.observation_space
        self.action_space = single_env.action_space
        self.max_episode_steps = getattr(single_env, "max_episode_steps", None)
        self.render_env = make_render_env()

    def rand_act(self):
        return torch.rand((self.cfg.num_envs, *self.action_space.shape)) * 2 - 1
    
    def reset(self):
        # Delegate the reset call to the underlying vectorized environment.
        return self.env.reset()

    def step_wait(self):
        # Delegate step_wait to the underlying vectorized environment.
        return self.env.step_wait()
    
    def step(self, action):
        return self.env.step(action)

    def get_state(self):
        # Call the custom get_state method on each sub-environment using env_method.
        # This returns a list of numpy arrays, one per environment.
        states = self.env.env_method("get_state")
        # Stack the states along a new axis and convert to a torch tensor.
        return torch.from_numpy(np.stack(states))

    def render(self, *args, **kwargs):
        # For rendering, call render on each sub-environment and return the first result.
        renders = self.env.env_method("render", *args, **kwargs)
        return renders[0]
