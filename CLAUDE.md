# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BFM-Zero Deployment Stack: deploying Behavioral Foundation Model (BFM-Zero) policies on the Unitree G1 humanoid robot. Supports sim-to-sim (MuJoCo) testing and real-robot deployment on Jetson Orin. 29-DOF humanoid with 50 Hz control loop.

## Key Commands

### Environment Setup
```bash
pip install -r requirements.txt
python download_hf_model.py --token <HF_TOKEN>  # downloads ONNX model + latent vars to model/
```

### Running Simulation (requires two terminals)
```bash
# Terminal 1: MuJoCo simulator
python -m sim_env.base_sim --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof.yaml

# Terminal 2: Policy (pick one)
./rl_policy/tracking.sh   # motion tracking
./rl_policy/reward.sh     # reward inference
./rl_policy/goal.sh       # goal reaching
```

### Direct Policy Invocation
```bash
python rl_policy/bfm_zero.py \
  --robot_config config/robot/g1.yaml \
  --policy_config config/policy/motivo_newG1.yaml \
  --model_path ./model/exported/FBcprAuxModel.onnx \
  --task config/exp/tracking/walking.yaml
```

### Real Robot
Use `config/robot/g1_real.yaml` instead of `g1.yaml`. Requires Unitree SDK2 with CycloneDDS on the Jetson Orin.

## Architecture

### Communication Model
Simulator and policy run as **separate processes** communicating via ZMQ pub-sub:
- **Port 5590**: Simulator publishes `LowStateMessage` (robot state)
- **Port 5591**: Policy publishes `LowCmdMessage` (joint commands)
- Message formats defined in `utils/common.py`

### Core Modules

- **`rl_policy/bfm_zero.py`** — Main policy loop (`BFMZeroPolicy`). Loads ONNX model, builds observations from config, runs 50 Hz scheduled inference, sends joint commands.
- **`sim_env/base_sim.py`** — MuJoCo simulator wrapper. Loads XML scene, runs physics at sim_dt=0.005, publishes state via ZMQ.
- **`sim_env/utils/simulation_bridge.py`** — ZMQ bridge translating between MuJoCo state and policy message formats. Applies PD control torques.
- **`rl_policy/observations/`** — Observation registry system. Base class in `base.py` uses `__init_subclass__()` to auto-register. Concrete observations (dof_pos, dof_vel, projected_gravity, etc.) in `bfm_zero.py`. Config references observations by string name.
- **`rl_policy/utils/state_processor.py`** — Subscribes to ZMQ state, maps Unitree joint names to Isaac ordering, maintains qpos/qvel.
- **`rl_policy/utils/command_sender.py`** — Publishes joint commands via ZMQ (sim) or unitree_sdk2 (real robot).
- **`utils/`** — ONNX inference wrapper (`onnx_module.py`), quaternion math (`math.py`), joint name mappings (`strings.py`), ZMQ port constants (`common.py`).

### Task Types (config-driven, no code changes needed)
- **tracking**: Uses discounted context window averaging over pre-computed latent z-variables from .pkl files
- **reward**: Cycles through reward-specific z-variables selected by name filter
- **goal**: Selects from pre-computed goal states (walking, dancing, fighting, etc.)

### Key Patterns
- **Joint ordering**: Unitree 29-DOF names mapped to Isaac convention via regex matching in `utils/strings.py:resolve_matching_names_values()`
- **Quaternion format**: (w, x, y, z) throughout
- **Observation vector**: Concatenation of current state + 4-step history (dof_pos, dof_vel, ang_vel, gravity, prev_actions) + 256-D latent z
- **Action pipeline**: Policy output → per-joint scaling → rescale (×5) → clipping → PD control (KP/KD from config)

### Configuration Hierarchy
```
config/
├── robot/     # g1.yaml (sim) vs g1_real.yaml (real) — joint limits, network interface
├── policy/    # motivo_newG1.yaml — observation groups, action scaling, PD gains, defaults
├── scene/     # g1_29dof.yaml — MuJoCo scene file, simulation timesteps
└── exp/       # Task configs: tracking/, reward/, goal/ — z-variable paths, parameters
```

### Interactive Controls (keyboard)
`]` enable policy, `[` start tracking, `n` switch reward/goal, `i` init pose, `o` stop

## Latent Z Inference (sister repo: `../BFM-Zero_inf`)

The `.pkl` files used by the three task types contain pre-computed 256-D latent z vectors. To generate z from your own motion data, use the inference repo at `../BFM-Zero_inf`.

### Setup
```bash
cd ../BFM-Zero_inf
pip install -r requirements.txt                        # torch, mujoco, safetensors, mujoco_warp
python download_hf_model.py --token <HF_TOKEN>         # downloads PyTorch checkpoint to ./model/checkpoint/model
jupyter notebook inference_tutorial.ipynb               # interactive tutorial
```

### Core Inference Model
`bfm_zero_inference_code/fb_cpr_aux/model.py` → `FBcprAuxModel` (extends `FBModel` from `fb/model.py`).

Key components:
- **`backward_map(obs)`**: Observation → 256-D latent z (the core encoder)
- **`forward_map(obs, z, action)`**: Predicts next observation
- **`act(obs, z)`**: Policy inference, (obs, z) → 29-D action
- **`project_z(z)`**: Projects z onto hypersphere (norm = sqrt(256) = 16)

### Three Inference Modes

All three use `backward_map` as core encoder, differing only in post-processing:

| Mode | Method | Input | Output |
|------|--------|-------|--------|
| **Tracking** | `model.tracking_inference(next_obs)` | Motion sequence → per-frame obs | z sequence `[T, 256]` (sliding window average over `seq_length` frames) |
| **Goal** | `model.goal_inference(next_obs)` | Single target pose → obs | Single z `[256]` |
| **Reward** | `model.reward_wr_inference(next_obs, rewards)` | Obs batch + scalar rewards | Single z `[256]` (soft-max weighted) |

### Pipeline: Motion File → Latent Z → Deploy

```
1. Prepare motion data (dof_positions, dof_velocities, root_pos/quat/vel/ang_vel per frame)
                                    ↓
2. MuJoCoBFMZeroEnv (env.py): set_state() per frame → _create_observation_backward() → next_obs dict
   Obs format: { state(64D), history_actor(372D), last_action(29D), privileged_state(463D) }
                                    ↓
3. model.backward_map(next_obs) → raw z, then aggregate per task type
                                    ↓
4. joblib.dump(z, "my_motion.pkl") → copy to deploy repo model/{tracking,reward,goal}_inference/
                                    ↓
5. Update config/exp/ yaml to point to new .pkl file
```

### Reward Functions (`bfm_zero_inference_code/inference/rewards.py`)
Uses MuJoCo Warp (GPU-accelerated batch evaluation):
- `MJWLocomotionReward`: Forward/backward/sidestep at target speed (`move_speed`, `move_angle`)
- `MJWRotationReward`: Spin around axis (`target_ang_velocity`)
- `MJWArmsReward` / `MJWMoveArmsReward` / `MJWSpinArmsReward`: Arm pose targets
- `MJWSitOnGroundReward`: Sitting posture

### Motion Data Format
Example file: `example_motion/dance1_subject2_50_jpos.npz` with fields:
- `body_positions`: `[T, num_bodies, 3]` — body positions per frame
- `body_rotations`: `[T, num_bodies, 4]` — body quaternions `[w,x,y,z]`
- `dof_positions`: `[T, 29]` — joint angles (Isaac ordering)
- `fps`: frame rate (typically 50)

### ONNX Export
The notebook also demonstrates exporting the PyTorch model to ONNX for use in the deploy repo:
```python
export_meta_policy_as_onnx(model, output_dir, "model.onnx", example_input, z_dim=256, history=True)
```
The exported ONNX model is what `rl_policy/bfm_zero.py` loads for real-time inference.
