# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BFM-Zero is a behavioral foundation model for humanoid robot control using unsupervised reinforcement learning. It trains a latent-conditioned policy on the Unitree G1 robot (29-DOF) using Forward-Backward (FB) representations with CPR (Critic Pessimism Regularization). Supports Isaac Sim and MuJoCo simulators.

## Commands

### Install dependencies
```bash
uv sync
```

### Training
```bash
uv run python -m humanoidverse.train
```
Change `buffer_device` to `"cuda:0"` if you have large vRAM.

### Inference (all use tyro CLI)
```bash
uv run python -m humanoidverse.tracking_inference --model_folder /path/to/model --data_path humanoidverse/data/lafan_29dof.pkl
uv run python -m humanoidverse.goal_inference --model_folder /path/to/model
uv run python -m humanoidverse.reward_inference --model_folder /path/to/model
```
Add `--simulator mujoco` to run without Isaac Sim. Add `--no-headless` for GUI, `--save_mp4` for video output.

### Linting
```bash
uv run ruff check .
uv run ruff format .
```
Ruff config: line-length 140, ignores E402 (lazy imports) and E731 (lambda assignments), includes import sorting (I).

## Architecture

### Entry Points
- `humanoidverse/train.py` — Training loop. Uses `tyro` for CLI config via `TrainConfig` (pydantic model). Orchestrates env creation, buffer management, agent updates, and evaluation callbacks.
- `humanoidverse/tracking_inference.py` — Motion tracking inference, exports ONNX.
- `humanoidverse/goal_inference.py` — Goal-reaching inference.
- `humanoidverse/reward_inference.py` — Reward-based task inference.

### Agent System (`humanoidverse/agents/`)
- `fb_cpr/` — FB-CPR agent: Forward-Backward representations with Critic Pessimism Regularization. Primary agent type.
- `fb_cpr_aux/` — FB-CPR with auxiliary losses variant.
- `fb/` — Base Forward-Backward agent (parent of fb_cpr).
- `buffers/` — Replay buffers: `DictBuffer` (transition-level) and `TrajectoryDictBufferMultiDim` (trajectory-level).
- `envs/humanoidverse_isaac.py` — Gymnasium wrapper bridging agents to the HumanoidVerse environment. Handles observation/action mapping and Isaac-specific expert trajectory loading.
- `evaluations/` — Evaluation callbacks (tracking eval via EMD metric).
- `nn_models.py` — Neural network building blocks (MLP, encoders). `nn_filters.py` / `nn_filter_models.py` — observation filtering/normalization.

### Environment System (`humanoidverse/envs/`)
- `legged_robot_motions/` — Main environment for motion-conditioned locomotion tasks.
- `legged_base_task/` — Base class for legged robot environments.
- `base_task/` — Root environment base class.
- `g1_env_helper/` — G1-specific environment configuration helpers.

### Simulator Backends (`humanoidverse/simulator/`)
- `isaacsim/` — Isaac Sim/Isaac Lab backend (Linux only).
- `mujoco/` — MuJoCo backend (cross-platform).
- `base_simulator/` — Abstract simulator interface.

### Configuration
Hydra-based YAML configs in `humanoidverse/config/`. The experiment config `config/exp/bfm_zero/bfm_zero.yaml` composes defaults for env, simulator, domain randomization, rewards, robot, terrain, and observations. Training entry point (`train.py`) uses `tyro` + pydantic for its own config, while the HumanoidVerse env uses Hydra internally via `hydra_overrides`.

### Data
Motion data in `humanoidverse/data/` (Git LFS): `lafan_29dof_10s-clipped.pkl` for training, `lafan_29dof.pkl` for evaluation.

### Key Design Patterns
- Agent configs use pydantic discriminated unions (`FBcprAgentConfig | FBcprAuxAgentConfig`) selected via the `name` field.
- The training loop is a single flat function `train_bfm_zero()` — not class-based.
- Observations/actions flow through the gymnasium wrapper in `agents/envs/` which translates between the agent interface and the HumanoidVerse env.
