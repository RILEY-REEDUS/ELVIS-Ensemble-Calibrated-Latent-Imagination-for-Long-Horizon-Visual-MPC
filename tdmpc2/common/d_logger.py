import dataclasses
import os
import datetime
import re
import imageio

import numpy as np
import pandas as pd
from termcolor import colored
from pathlib import Path


CONSOLE_FORMAT = [
	("iteration", "I", "int"),
	("episode", "E", "int"),
	("step", "I", "int"),
	("episode_reward", "R", "float"),
	("episode_success", "S", "float"),
	("total_time", "T", "time"),
	("pi_loss", "L_pi", "float"),
	("value_loss", "L_v", "float"),
	("reward_loss", "L_r", "float"),
	("consistency_loss", "L_c", "float"),
	("observation_loss", "L_o", "float"),
	("kl_b_loss", "KL_b", "float"),
	("kl_z_loss", "KL_z", "float"),
	("vae_loss", "L_vae", "float"),
	("v_head_loss", "L_vh", "float"),
	("action_head_loss", "L_a", "float"),
]

CAT_TO_COLOR = {
	"pretrain": "yellow",
	"train": "blue",
	"eval": "green",
}

def make_dir(dir_path):
	"""Create directory if it does not already exist."""
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def print_run(cfg):
	"""
	Pretty-printing of current run information.
	Logger calls this method at initialization.
	"""
	prefix, color, attrs = "  ", "green", ["bold"]

	def _limstr(s, maxlen=36):
		return str(s[:maxlen]) + "..." if len(str(s)) > maxlen else s

	def _pprint(k, v):
		print(
			prefix + colored(f'{k.capitalize()+":":<15}', color, attrs=attrs), _limstr(v)
		)

	observations  = ", ".join([str(v) for v in cfg.obs_shape.values()])
	kvs = [
		("task", cfg.task_title),
		("steps", f"{int(cfg.steps):,}"),
		("observations", observations),
		("actions", cfg.action_dim),
		("experiment", cfg.exp_name),
	]
	w = np.max([len(_limstr(str(kv[1]))) for kv in kvs]) + 25
	div = "-" * w
	print(div)
	for k, v in kvs:
		_pprint(k, v)
	print(div)


def cfg_to_group(cfg, return_list=False):
	"""
	Return a wandb-safe group name for logging.
	Optionally returns group name as list.
	"""
	lst = [cfg.task, re.sub("[^0-9a-zA-Z]+", "-", cfg.exp_name)]
	return lst if return_list else "-".join(lst)


class VideoRecorder:
	"""Utility class for logging evaluation videos."""

	def __init__(self, cfg, wandb, fps=15):
		self.cfg = cfg
		self._save_dir = make_dir(cfg.work_dir / 'eval_video')
		self._wandb = wandb
		self.fps = fps
		self.frames = []
		self.enabled = False

	def init(self, env, enabled=True):
		self.frames = []
		self.enabled = self._save_dir and enabled
		self.record(env)

	def record(self, env):
		if self.enabled:
			self.frames.append(env.render())

	def save(self, step, key='videos/eval_video'):
		if self.enabled and len(self.frames) > 0:
			frames = np.stack(self.frames)
			return self._wandb.log(
				{key: self._wandb.Video(frames.transpose(0, 3, 1, 2), fps=self.fps, format='mp4')}, step=step
			)

class LocalVideoRecorder:
    """Utility class for saving evaluation videos locally."""

    def __init__(self, cfg, fps=15):
        self.cfg = cfg
        self._save_dir = Path(cfg.work_dir) / 'eval_video'
        self._save_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
        self.fps = fps
        self.frames = []
        self.enabled = False

    def init(self, env, enabled=True):
        """Initialize the video recorder for a new episode."""
        self.frames = []
        self.enabled = enabled
        self.record(env)

    def record(self, env):
        """Record a single frame from the environment."""
        if self.enabled:
            frame = env.render()
            self.frames.append(frame)

    def save(self, step, filename=None):
        """Save the recorded frames as a video file."""
        if self.enabled and len(self.frames) > 0:
            if filename is None:
                filename = f"eval_video_{step}.mp4"
            
            video_path = str(self._save_dir / filename)
            
            # Alternatively, save with imageio (better compatibility)
            imageio.mimsave(video_path, self.frames, fps=self.fps)

            print(f"Saved evaluation video: {video_path}")


class DLogger:
	"""Primary logging object. Logs either locally or using wandb."""

	def __init__(self, cfg):
		self._log_dir = make_dir(cfg.work_dir)
		self._model_dir = make_dir(self._log_dir / "models")
		self._save_csv = cfg.save_csv
		self._save_agent = cfg.save_agent
		self._group = cfg_to_group(cfg)
		self._seed = cfg.seed
		self._eval = []
		self._train = []
		self._train_extras = []
		self._video = LocalVideoRecorder(cfg)
		print_run(cfg)
		self.project = cfg.get("wandb_project", "none")
		self.entity = cfg.get("wandb_entity", "none")
		if not cfg.enable_wandb or self.project == "none" or self.entity == "none":
			print(colored("Wandb disabled.", "blue", attrs=["bold"]))
			cfg.save_agent = False
			cfg.save_video = True
			self._wandb = None
			return
		os.environ["WANDB_SILENT"] = "true" if cfg.wandb_silent else "false"
		import wandb

		wandb.init(
			project=self.project,
			entity=self.entity,
			name=str(cfg.seed),
			group=self._group,
			tags=cfg_to_group(cfg, return_list=True) + [f"seed:{cfg.seed}"],
			dir=self._log_dir,
			config=dataclasses.asdict(cfg),
		)
		print(colored("Logs will be synced with wandb.", "blue", attrs=["bold"]))
		self._wandb = wandb
		self._video = LocalVideoRecorder(cfg)

	@property
	def video(self):
		return self._video

	@property
	def model_dir(self):
		return self._model_dir

	def save_agent(self, agent=None, replay=None, identifier='final'):
		if self._save_agent and agent:
			fp = self._model_dir / f'{str(identifier)}.pt'
			agent.save(fp)
			if self._wandb:
				artifact = self._wandb.Artifact(
					self._group + '-' + str(self._seed) + '-' + str(identifier),
					type='model',
				)
				artifact.add_file(fp)
				self._wandb.log_artifact(artifact)
		if replay:
			fp = self._model_dir / f'{str(identifier)}.replay.pt'
			replay.save(fp)

	def finish(self, agent=None):
		try:
			self.save_agent(agent)
		except Exception as e:
			print(colored(f"Failed to save model: {e}", "red"))
		if self._wandb:
			self._wandb.finish()

	def _format(self, key, value, ty):
		if ty == "int":
			return f'{colored(key+":", "blue")} {int(value):,}'
		elif ty == "float":
			return f'{colored(key+":", "blue")} {value:.01f}'
		elif ty == "time":
			value = str(datetime.timedelta(seconds=int(value)))
			return f'{colored(key+":", "blue")} {value}'
		else:
			raise f"invalid log format type: {ty}"

	def _print(self, d, category):
		category = colored(category, CAT_TO_COLOR[category])
		pieces = [f" {category:<14}"]
		for k, disp_k, ty in CONSOLE_FORMAT:
			if k in d:
				pieces.append(f"{self._format(disp_k, d[k], ty):<22}")
		print("   ".join(pieces))

	def log(self, d, category="train"):
		assert category in CAT_TO_COLOR.keys(), f"invalid category: {category}"
		if self._wandb:
			if category in {"train", "eval"}:
				xkey = "step"
			elif category == "pretrain":
				xkey = "iteration"
			_d = dict()
			for k, v in d.items():
				_d[category + "/" + k] = v
			self._wandb.log(_d, step=d[xkey])
		if category == "eval" and self._save_csv:
			keys = ["step", "episode_reward"]
			self._eval.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "eval.csv", header=keys, index=None
			)
		if category == "train" and self._save_csv:
			#NOTE: self._eval should not be empty when running this. Now this
			# requires running an eval beforehand, to be fixed.
			keys = ["step", "episode_reward"]
			self._train.append(np.array([d[keys[0]], d[keys[1]]]))
			pd.DataFrame(np.array(self._eval)).to_csv(
				self._log_dir / "train.csv", header=keys, index=None
			)
			if "pi_loss" in d.keys(): # dmpc
				keys = ["pi_loss", "value_loss", "reward_loss", "kl_z_loss", "vae_loss", "action_head_loss"]
				self._train_extras.append(np.array([d[keys[i]] for i in range(len(keys))]))
				pd.DataFrame(np.array(self._train_extras)).to_csv(
						self._log_dir / "train_extra.csv", header=keys, index=None
			)
		self._print(d, category)
