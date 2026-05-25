> 基于[LeCAR-Lab BFM-Zero](https://github.com/LeCAR-Lab/BFM-Zero)项目修改

[[arXiv]](https://arxiv.org/abs/2511.04131)
[[Paper]](https://lecar-lab.github.io/BFM-Zero/resources/paper.pdf)
[[Website]](https://lecar-lab.github.io/BFM-Zero/)

> **Sim2Sim / Sim2Real 部署代码请见 [`deploy`](https://github.com/kzzzza/BFM-Zero/tree/deploy) 分支。**

## 项目简介

BFM-Zero 是一个面向人形机器人控制的**行为基础模型**（Behavioral Foundation Model），通过**无监督强化学习**在 Unitree G1（29 自由度）上训练一个 latent 条件化策略。核心技术为 Forward-Backward 表征 + CPR（Critic Pessimism Regularization）。

本分支（`main`）提供完整的**训练 + 评测 + 推理流水线**，同时支持 Isaac Sim 与 MuJoCo 两个仿真后端。

## 环境要求

- Python 3.10
- 支持 CUDA 的 GPU
- Isaac Sim（仅 Linux）或 MuJoCo（跨平台）

## 安装

### 1. 克隆仓库并下载模型数据

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git
cd BFM-Zero
```

> 从 HuggingFace 下载数据：<https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/data>

### 2. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

或者用 pip 安装：`pip install uv`

### 3. 安装依赖

在 `BFM-Zero` 根目录下执行：

```bash
uv sync
```

## 数据

- **动作数据**：一并下载到 `humanoidverse/data/`：
  - `lafan_29dof_10s-clipped.pkl`：训练用动作数据。
  - `lafan_29dof.pkl`：评测用动作数据。
- 数据格式可参考 [Issue #12](https://github.com/LeCAR-Lab/BFM-Zero/issues/12) 中的讨论。

## 训练

### 启动

```bash
uv run python -m humanoidverse.train
```

### 主要参数

训练入口位于 `humanoidverse/train.py` 中的 `train_bfm_zero()`，由 `TrainConfig`（pydantic 模型）驱动。主要参数按领域分类如下：

| 类别 | 参数 |
|------|------|
| **规模 (Scale)** | `num_env_steps`（默认 30M）、`online_parallel_envs`（默认 16）、`buffer_size`（默认 500k）、`checkpoint_every_steps`（默认 5M） |
| **路径 (Paths)** | `work_dir`、`motions` / `motions_root`（专家动作数据） |
| **运行 (Run)** | `seed`、`use_wandb`、`wandb_pname` / `wandb_gname` / `wandb_ename` |
| **策略 / 优化器 (Policy)** | `update_agent_every`（默认 500）、`num_agent_updates`（默认 50）、`num_seed_steps`（默认 50k）；agent 内部 `batch_size`（默认 512）、`lr_actor` / `lr_critic`（默认 1e-4）、`discount`（默认 0.98） |
| **机器人 / 环境** | 通过 `hydra_overrides` 覆盖（如 `robot=...`、`robot.control.action_scale=...`、`env.config.lie_down_init=...`） |

如需自定义参数，可在代码中传入自定义的 `TrainConfig`，或扩展 CLI 以接受 Hydra / tyro 覆盖。

**经验提示**：训练 50–100M 步后，`eval/emd` 应低于 0.75。

<div align="center">
<img src="static/images/training_curve.png" style="height:300px;" />
</div>

## 单卡 4090 训练

默认配置已针对单卡 RTX 4090（24 GB 显存）做了精简，无需额外调整即可直接启动：

| 项 | 原值 | 4090 友好默认值 |
|----|------|----------------|
| 网络隐藏层维度 `hidden_dim` | 1024 | **512** |
| 训练 `batch_size` | 1024 | **512** |
| 判别器隐藏层数 | 3 | **2** |
| 评测 `num_envs` | 1024 | **512** |
| `disable_tqdm` | True | **False** |

此外，在 48GB 的魔改4090 上请将 `buffer_device` 设为 `"cuda"`（默认值；显存更大的机器，例如 ≥ 48 GB，可改回 `"cuda:0"` 并相应放大 `hidden_dim` / `batch_size`）。

---

## 推理

训练完成后，提供三个推理脚本（均使用 [tyro](https://github.com/brentyi/tyro) 进行 CLI 解析）：

| 脚本 | 用途 |
|------|------|
| **`humanoidverse.tracking_inference`** | 动作追踪 → 提取 latent \(z\)，导出 ONNX |
| **`humanoidverse.goal_inference`** | 目标到达 → 为不同目标计算 \(z\) |
| **`humanoidverse.reward_inference`** | 奖励驱动任务 → 计算 \(z\) 并评测表现 |

### 使用方法参考

```bash
uv run python -m humanoidverse.tracking_inference --help
uv run python -m humanoidverse.goal_inference --help
uv run python -m humanoidverse.reward_inference --help
```

**通用参数：**

- `--model_folder`：训练好的模型目录（需包含 `checkpoint/` 与 `config.json`）。
- `--data_path`（可选）：覆盖默认的 LaFan 数据路径。
- `--simulator`：`isaacsim`（默认）或 `mujoco`。**指定 `--simulator mujoco` 可在没有 Isaac Lab 的机器上运行**（仅 MuJoCo，可直接用于 sim2sim 可视化）。
- `--headless`（默认 `True`）：无 GUI 运行；加 `--no-headless` 打开 viewer。
- `--save_mp4`：保存渲染视频。

**输出：** 所有推理脚本会把策略导出为 ONNX（`{model_name}.onnx`），落在 `exported/` 下各自的子目录里。Tracking 推理同时会通过 `humanoidverse/utils/helpers.py` 中的 `export_backward_map_as_onnx()` 导出 backward map 的 ONNX。

---

### Tracking inference

运行动作追踪、导出 ONNX，并可选地保存专家 vs 策略的对比视频。

```bash
uv run python -m humanoidverse.tracking_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --no-headless \
    --save_mp4
```

- `--model_folder` 指向 **外层** 模型目录（包含 `checkpoint/` 的那一级）。
- 可在脚本内通过 `--motion_list` 指定追踪哪些动作 ID（默认 `[25]`）。

**输出**（`model_folder/tracking_inference/` 下）：

- `zs_{MOTION_ID}.pkl`：每段动作对应的 latent \(z\)。
- `tracking.mp4`：专家 vs 策略对比视频（启用 `--save_mp4` 时）。

---

### Goal inference

为预定义的目标计算 \(z\)，并可选地渲染目标到达视频。

```bash
uv run python -m humanoidverse.goal_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --save_mp4
```

- 遍历预定义的一组目标并计算对应的 \(z\)。
- 依赖 `goal_frames_lafan29dof.json`（脚本会在多个位置查找）。

**输出**（`model_folder/goal_inference/` 下）：

- `goal_reaching.pkl`：`{goal_name -> z}` 字典。
- `videos/*.mp4`：每个目标对应的视频（启用 `--save_mp4` 时）。

---

### Reward inference

运行奖励驱动的任务推理：计算 \(z\)，并可选地跑 rollout 做评测。

```bash
uv run python -m humanoidverse.reward_inference \
    --model_folder /path/to/model \
    --save_mp4
```

**关键参数：**

| 参数 | 说明 |
|------|------|
| `--num_samples` | 单次推理时缓冲区中的样本数量（默认 150000）。 |
| `--n_inferences` | 每个奖励任务的推理 latent 数量（默认 1）。 |
| `--episode_length` | 每次 rollout 的步数（默认 500）。 |
| `--skip_rollouts` | 仅计算 \(z\)，不跑可视化 rollout。 |

**输出**（`model_folder/reward_inference/` 下）：

- `reward_locomotion.pkl`：`{task_name -> z}` 字典。
- `videos/*.mp4`：每个任务对应的视频（启用 `--save_mp4` 时）。

---

## Roadmap

- [ ] **Unitree R1 训练支持**：R1 URDF 与网格资产已加入 `humanoidverse/data/robots/r1/`，对应训练 config 仍在补齐中。

## Lint

```bash
uv run ruff check .
uv run ruff format .
```

## License

BFM-Zero 采用 CC BY-NC 4.0 协议授权，详见 [LICENSE](LICENSE)。

## Citation

```bibtex
@misc{li2025bfmzeropromptablebehavioralfoundation,
      title={BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning}, 
      author={Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
      year={2025},
      eprint={2511.04131},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.04131}, 
}
```

