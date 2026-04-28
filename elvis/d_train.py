import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.DreamBuffer import DreamBuffer
from envs import make_env
from trainer.d_online_trainer import DOnlineTrainer
from common.d_logger import DLogger
from DMPC import DMPC

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.dynamic_shapes = True
torch._dynamo.config.assume_static_by_default = False
torch._dynamo.config.compiled_autograd = True


@hydra.main(config_name='d_config', config_path='.')
def train(cfg: dict):
	"""
	Train a single-task online belief-MPC agent.

	Example:
		$ python d_train.py task=reacher_hard
	"""
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	trainer = DOnlineTrainer(
		cfg=cfg,
		env=make_env(cfg),
		agent=DMPC(cfg),
		buffer=DreamBuffer(cfg),
		logger=DLogger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
