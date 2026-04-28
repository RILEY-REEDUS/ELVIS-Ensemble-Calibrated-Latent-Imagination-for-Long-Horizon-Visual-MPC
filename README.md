# ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC

Reference implementation accompanying the paper **"ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC"** (RSS 2026).

ELVIS trains a Dreamer-style recurrent state-space model (RSSM) world model and uses **parallel multi-modal MPPI** for planning over the learned latent dynamics. The planner's effective horizon is **calibrated by Q-ensemble disagreement**: per-step λ-returns are scheduled by ensemble uncertainty, so the agent relies less on long-horizon predictions when the value function is uncertain about them. This lets the planner exploit deep imagination when the world model is trustworthy and gracefully shorten its horizon when it is not.

## Method Overview

The agent maintains:

- **A categorical RSSM world model** that predicts the next belief state from `(b_{t-1}, z_{t-1}, a_{t-1})` and decodes a stochastic latent `z_t` from observations. Belief and latent are trained jointly with a free-bit KL term.
- **A reward head and a Q-value ensemble** trained on imagined rollouts under the learned dynamics. Bootstrapping uses a soft λ-return, with **per-step λ scheduled by ensemble disagreement** so that the planner relies less on uncertain long-horizon predictions.
- **A Gaussian policy prior** trained on the same rollouts to provide warm-start trajectories during planning.

At each environment step, planning runs CEM-style updates over `G` parallel modes (an explicit multi-modal proposal), each with its own per-timestep mean and standard deviation. Modes mix policy-rollout trajectories with Gaussian samples in mode-dependent proportions for diversity. Mode-and-elite sampling with Gumbel-softmax selects the executed action, recovering both exploitation of the best mode and stochasticity for exploration.

For the full algorithmic and theoretical details see the paper.

## Repository Layout

```
elvis/
├── DMPC.py                       # agent: training loop, MPPI planner, AC losses
├── d_train.py                    # entry point (Hydra)
├── d_config.yaml                 # default hyperparameters
├── common/
│   ├── DreamWorldModel.py        # RSSM + reward + Q-ensemble + policy
│   ├── DreamBeliefTracker.py     # belief GRU, posterior/prior, decoders
│   ├── DreamBuffer.py            # episodic replay buffer
│   ├── d_layers.py               # encoders, decoders, MLPs, ensembles
│   ├── d_logger.py, math.py, scale.py, init.py, tools.py, parser.py, seed.py
├── envs/
│   ├── dmcontrol.py              # DMControl wrapper (state and pixel)
│   ├── tasks/                    # custom DMControl task variants
│   └── wrappers/                 # tensor / pixel / vectorized env wrappers
└── trainer/
    ├── d_online_trainer.py       # online single-task training loop
    └── base.py
```

## Installation

```bash
conda env create -f docker/environment.yaml
conda activate <env-name>
```

A reference Dockerfile is provided in `docker/`.

## Training

Single-task online training on DMControl with pixel observations. Default task is `reacher_hard`.

Run from the repository root with the environment's Python interpreter and the training script's full path:

```bash
<path/to/python> <path/to/repo>/elvis/d_train.py task=walker_walk seed=1
<path/to/python> <path/to/repo>/elvis/d_train.py task=cheetah_run seed=1 num_envs=4
```

Training writes checkpoints, CSV logs, and (optionally) videos to `logs/<task>/<seed>/<exp_name>/`.

Useful flags:

| Flag                          | Default        | Notes                                                |
| ----------------------------- | -------------- | ---------------------------------------------------- |
| `task`                        | `reacher_hard` | DMControl task name                                  |
| `num_envs`                    | `1`            | Vectorized envs                                       |
| `steps`                       | `500_000`      | Total environment steps                              |
| `compile`                     | `true`         | `torch.compile` on hot loops (PyTorch nightly)       |
| `save_video`                  | `true`         | Save eval rollouts                                   |
| `num_gmms`                    | `10`           | Number of parallel MPPI modes                        |
| `iterations`                  | `6`            | CEM iterations per planning step                     |
| `num_samples` / `num_elites`  | `64` / `8`     | MPPI samples and elites **per mode**                 |
| `plan_horizon` / `imag_horizon` | `16` / `16`  | Planning and imagination horizons                    |

See `elvis/d_config.yaml` for the full list.

## Resuming

Training auto-resumes if a numeric checkpoint pair `<step>.pt` and `<step>.replay.pt` exists in the work dir; pass an explicit path via `resume_fp_agent=<path>` to override.

## Citation

```bibtex
@inproceedings{elvis2026,
  title     = {ELVIS: Ensemble-Calibrated Latent Imagination for Long-Horizon Visual MPC},
  author    = {Yurui Du, Pinhao Song, Yutong Hu, Renaud Detry},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2026},
}
```

The implementation builds on prior open-source work; see `LICENSE` for licensing and the references in the paper for full credit:

- TD-MPC2 (Hansen et al., 2024) — planning-with-value structure
- DreamerV3 (Hafner et al., 2024) — RSSM world model design
- TensorDict / TorchRL (Meta) — `torch.compile`-friendly tensor utilities

## License

This project is released under the Apache-2.0 License — see `LICENSE`.
