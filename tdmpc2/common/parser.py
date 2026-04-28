import dataclasses
import os
import re
from pathlib import Path
from typing import Any

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from common import MODEL_SIZE


def cfg_to_dataclass(cfg, frozen=False):
	"""
	Convert an OmegaConf config to a dataclass — prevents graph breaks under torch.compile.
	"""
	cfg_dict = OmegaConf.to_container(cfg)
	fields = []
	for key, value in cfg_dict.items():
		fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
	dataclass = dataclasses.make_dataclass("Config", fields, frozen=frozen)
	def get(self, val, default=None):
		return getattr(self, val, default)
	dataclass.get = get
	return dataclass()


def get_project_root_fallback() -> Path:
	if HydraConfig.initialized():
		return Path(hydra.utils.get_original_cwd())
	return Path(os.getcwd())


def parse_cfg(cfg: OmegaConf) -> OmegaConf:
	"""
	Parse a Hydra config: substitute None→True for missing flags, evaluate small algebraic
	expressions, set work_dir, and apply model-size presets.
	"""
	OmegaConf.set_struct(cfg, False)

	# None → True
	for k in cfg.keys():
		try:
			if cfg[k] is None:
				cfg[k] = True
		except Exception:
			pass

	# Small algebraic expressions (e.g. "256+128")
	for k in cfg.keys():
		try:
			v = cfg[k]
			if isinstance(v, str):
				match = re.match(r"(\d+)([+\-*/])(\d+)", v)
				if match:
					cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
					if isinstance(cfg[k], float) and cfg[k].is_integer():
						cfg[k] = int(cfg[k])
		except Exception:
			pass

	# Convenience
	project_root = get_project_root_fallback()
	cfg.work_dir = project_root / 'logs' / cfg.task / str(cfg.seed) / cfg.exp_name
	cfg.task_title = cfg.task.replace("-", " ").title()
	cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins - 1)

	# Model size preset
	if cfg.get('model_size', None) is not None:
		assert cfg.model_size in MODEL_SIZE.keys(), \
			f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
		for k, v in MODEL_SIZE[cfg.model_size].items():
			cfg[k] = v

	# Single-task only
	cfg.multitask = False
	cfg.task_dim = 0
	cfg.tasks = [cfg.task]

	return cfg_to_dataclass(cfg)
