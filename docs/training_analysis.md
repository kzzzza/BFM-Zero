# BFM-Zero 训练流程分析

## 1. 训练入口与启动流程

### 入口点

- **文件**: `humanoidverse/train.py`
- **启动命令**: `uv run python -m humanoidverse.train`
- **入口函数**: `train_bfm_zero()` (第587行)

### 启动流程

```
train_bfm_zero()
  → 构建 TrainConfig (pydantic 配置模型)
  → cfg.build() → 创建 Workspace 对象
  → workspace.train() → workspace.train_online()
```

---

## 2. 训练流程详解

### 2.1 初始化阶段 (`Workspace.__init__`, 第188行)

1. **创建环境**: 根据 `HumanoidVerseIsaacConfig` 或其他配置构建训练环境，获取观察空间和动作空间
2. **创建工作目录**: `self.work_dir` 由 `cfg.work_dir` 指定（默认 `results/bfmzero-isaac`）
3. **保存配置**: 保存 `config.yaml`（Hydra配置）和 `config.json`（TrainConfig完整配置）到工作目录
4. **创建/加载 Agent**: 调用 `create_agent_or_load_checkpoint()`，若存在 checkpoint 则加载，否则新建
5. **初始化评估器**: 根据 `evaluations` 配置构建评估实例
6. **初始化 WandB**: 若 `use_wandb=True` 则初始化 wandb 日志
7. **初始化日志**: 创建 CSV logger 用于记录训练和评估指标

### 2.2 专家数据加载 (`train_online`, 第261行)

- **Isaac Sim 模式** (`load_isaac_expert_data=True`): 调用 `load_expert_trajectories_from_motion_lib()` 直接从 motion library 加载
- **离线数据模式**: 调用 `load_expert_trajectories()` 从磁盘加载 pickle 文件

### 2.3 Replay Buffer 创建 (第284行)

两种 buffer 类型:
- **TrajectoryDictBufferMultiDim** (`use_trajectory_buffer=True`): 轨迹级别存储，容量 = `buffer_size / online_parallel_envs`
- **DictBuffer**: 转换级别(transition)存储，容量 = `buffer_size`

若存在 checkpoint 的 buffer，会从 `checkpoint/buffers/train` 加载。

### 2.4 主训练循环 (第335行)

```
for t in range(checkpoint_time, num_env_steps, online_parallel_envs):
```

每次迭代处理 `online_parallel_envs` 个并行环境的一步交互。核心步骤：

#### (a) Checkpoint 保存
- 每 `checkpoint_every_steps` 步保存一次（默认 9,600,000 步）

#### (b) 评估
- 每 `eval_every_steps` 步执行一次评估（默认 9,600,000 步）
- 评估后更新优先级采样权重（若 `prioritization=True`）

#### (c) 动作选择
- **种子阶段** (`t < num_seed_steps`): 随机采样动作（默认前 10,240 步）
- **训练阶段**: Agent 根据当前观察和 latent context `z` 生成动作
  - Context `z` 通过 `agent.maybe_update_rollout_context()` 从 replay buffer 中采样更新

#### (d) 环境交互
- 执行动作 → 获取新观察、奖励、终止/截断信号
- 将 transition 或 trajectory 数据存入 replay buffer

#### (e) Agent 更新
- 每 `update_agent_every` 步（默认 512 步）执行 `num_agent_updates` 次更新（默认 16 次）
- 调用 `agent.update(replay_buffer, t)`

#### (f) 日志记录
- 每 `log_every_updates` 步（默认 384,000 步）记录训练指标
- 输出到控制台、CSV 文件，以及 WandB（若启用）

---

## 3. Checkpoint 与模型保存

### 保存位置

```
{work_dir}/checkpoint/
├── train_status.json          # 记录当前训练步数 {"time": <step>}
├── buffers/
│   └── train/                 # Replay buffer 数据（若 checkpoint_buffer=True）
└── ... (agent 模型文件)       # 由 agent.save() 保存的模型权重
```

- **默认 work_dir**: `results/bfmzero-isaac`
- **完整路径**: `results/bfmzero-isaac/checkpoint/`

### 保存逻辑 (`Workspace.save`, 第578行)

```python
def save(self, time, replay_buffer):
    self.agent.save(str(self.work_dir / CHECKPOINT_DIR_NAME))  # 保存模型
    if self.cfg.checkpoint_buffer:
        replay_buffer["train"].save(...)                        # 保存 buffer
    # 保存训练进度 (train_status.json)
```

### 恢复逻辑 (`create_agent_or_load_checkpoint`, 第166行)

启动时自动检测 `{work_dir}/checkpoint/` 是否存在：
- 存在 → 从 `train_status.json` 读取步数，加载模型和 buffer，从断点继续
- 不存在 → 从零开始训练

### 其他保存的文件

```
{work_dir}/
├── config.json                # TrainConfig 完整配置
├── config.yaml                # Hydra 环境配置（Isaac Sim 模式）
├── train_log.txt              # 训练指标 CSV
├── {eval_name}.csv            # 评估指标 CSV
└── checkpoint/                # 模型 checkpoint
```

---

## 4. 训练参数调整位置

### 4.1 顶层训练参数 (`TrainConfig`, 第71行)

在 `train_bfm_zero()` 函数中直接修改（第594行起）:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `work_dir` | `'results/bfmzero-isaac'` | 输出目录 |
| `seed` | `4728` | 随机种子 |
| `online_parallel_envs` | `512` | 并行环境数 |
| `num_env_steps` | `384,000,000` | 总训练步数 |
| `num_seed_steps` | `10,240` | 随机探索步数 |
| `update_agent_every` | `512` | 每N步更新一次 agent |
| `num_agent_updates` | `16` | 每次更新的梯度步数 |
| `checkpoint_every_steps` | `9,600,000` | Checkpoint 保存间隔 |
| `checkpoint_buffer` | `True` | 是否保存 replay buffer |
| `eval_every_steps` | `9,600,000` | 评估间隔 |
| `log_every_updates` | `384,000` | 日志记录间隔 |
| `buffer_size` | `2,560,000` | Replay buffer 容量 |
| `buffer_device` | `'cuda'` | Buffer 存储设备 |
| `use_trajectory_buffer` | `True` | 使用轨迹级 buffer |
| `disable_tqdm` | `False` | 是否关闭进度条 |

### 4.2 优先级采样参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `prioritization` | `True` | 启用优先级采样 |
| `prioritization_mode` | `'exp'` | 优先级模式 (bin/exp/lin) |
| `prioritization_min_val` | `0.5` | 优先级下限 |
| `prioritization_max_val` | `2.0` | 优先级上限 |
| `prioritization_scale` | `2.0` | 优先级缩放 |

### 4.3 Agent 参数 (`FBcprAuxAgentConfig`, 第596行)

#### 模型架构 (`FBcprAuxModelConfig`)

| 参数 | 值 | 说明 |
|------|-----|------|
| `device` | `'cuda'` | 模型设备 |
| `z_dim` | `256` | Latent 空间维度 |
| `norm_z` | `True` | 是否归一化 z |
| `seq_length` | `8` | 序列长度 |
| `actor_std` | `0.05` | Actor 标准差 |
| `amp` | `False` | 混合精度训练 |
| `inference_batch_size` | `500,000` | 推理批大小 |

#### 网络架构（各子网络均可独立调整）

- **Forward Network (`f`)**: `hidden_dim=2048, hidden_layers=6, model='residual', num_parallel=2`
- **Backward Network (`b`)**: `hidden_dim=256, hidden_layers=1, norm=True`
- **Actor**: `hidden_dim=2048, hidden_layers=6, model='residual'`
- **Critic**: `hidden_dim=2048, hidden_layers=6, model='residual', num_parallel=2`
- **Discriminator**: `hidden_dim=1024, hidden_layers=3`
- **Aux Critic**: `hidden_dim=2048, hidden_layers=6, model='residual', num_parallel=2`

每个子网络都有 `input_filter` 配置，控制哪些观察键被输入到该网络。

#### 观察归一化 (`obs_normalizer`)

使用 BatchNorm 对 `state`、`privileged_state`、`last_action`、`history_actor` 分别归一化，momentum=0.01。

#### 训练超参数 (`FBcprAuxAgentTrainConfig`, 第628行)

| 参数 | 值 | 说明 |
|------|-----|------|
| `batch_size` | `1024` | 训练批大小 |
| `discount` | `0.98` | 折扣因子 |
| `lr_f` | `3e-4` | Forward 网络学习率 |
| `lr_b` | `1e-5` | Backward 网络学习率 |
| `lr_actor` | `3e-4` | Actor 学习率 |
| `lr_critic` | `3e-4` | Critic 学习率 |
| `lr_discriminator` | `1e-5` | Discriminator 学习率 |
| `lr_aux_critic` | `3e-4` | Aux Critic 学习率 |
| `weight_decay` | `0.0` | 权重衰减 |
| `clip_grad_norm` | `0.0` | 梯度裁剪（0=不裁剪） |
| `fb_target_tau` | `0.01` | FB 目标网络 EMA 系数 |
| `critic_target_tau` | `0.005` | Critic 目标网络 EMA 系数 |
| `ortho_coef` | `100.0` | 正交正则化系数 |
| `train_goal_ratio` | `0.2` | 训练目标比例 |
| `fb_pessimism_penalty` | `0.0` | FB 悲观惩罚 |
| `actor_pessimism_penalty` | `0.5` | Actor 悲观惩罚 (CPR) |
| `critic_pessimism_penalty` | `0.5` | Critic 悲观惩罚 |
| `aux_critic_pessimism_penalty` | `0.5` | Aux Critic 悲观惩罚 |
| `stddev_clip` | `0.3` | 标准差裁剪 |
| `q_loss_coef` | `0.0` | Q-loss 系数 |
| `reg_coeff` | `0.05` | 正则化系数 |
| `reg_coeff_aux` | `0.02` | Aux 正则化系数 |
| `scale_reg` | `True` | 缩放正则化 |
| `relabel_ratio` | `0.8` | 重标注比例 |
| `expert_asm_ratio` | `0.6` | 专家 ASM 比例 |
| `use_mix_rollout` | `True` | 混合 rollout |
| `update_z_every_step` | `100` | Z 更新间隔 |
| `z_buffer_size` | `8192` | Z buffer 大小 |
| `rollout_expert_trajectories` | `True` | 使用专家轨迹 rollout |
| `rollout_expert_trajectories_length` | `250` | 专家 rollout 长度 |
| `rollout_expert_trajectories_percentage` | `0.5` | 专家 rollout 比例 |
| `grad_penalty_discriminator` | `10.0` | Discriminator 梯度惩罚 |
| `weight_decay_discriminator` | `0.0` | Discriminator 权重衰减 |

#### 辅助奖励 (`aux_rewards` & `aux_rewards_scaling`, 第664行)

| 奖励名 | 缩放系数 | 说明 |
|--------|---------|------|
| `penalty_action_rate` | -0.1 | 动作变化率惩罚 |
| `penalty_feet_ori` | -0.4 | 脚部方向惩罚 |
| `penalty_ankle_roll` | -4.0 | 踝关节翻转惩罚 |
| `limits_dof_pos` | -10.0 | 关节角度限位惩罚 |
| `penalty_slippage` | -2.0 | 滑动惩罚 |
| `penalty_undesired_contact` | -1.0 | 非期望接触惩罚 |
| `penalty_torques` | 0.0 | 力矩惩罚（未启用） |
| `limits_torque` | 0.0 | 力矩限制（未启用） |

### 4.4 环境参数 (`HumanoidVerseIsaacConfig`, 第671行)

| 参数 | 值 | 说明 |
|------|-----|------|
| `device` | `'cuda:0'` | 环境设备 |
| `lafan_tail_path` | `'humanoidverse/data/lafan_29dof_10s-clipped.pkl'` | 训练动作数据 |
| `relative_config_path` | `'exp/bfm_zero/bfm_zero'` | Hydra 配置路径 |
| `include_last_action` | `True` | 观察中包含上一步动作 |
| `include_history_actor` | `True` | 包含 actor 历史 |
| `root_height_obs` | `True` | 包含根节点高度观察 |
| `disable_obs_noise` | `False` | 观察噪声 |
| `disable_domain_randomization` | `False` | 域随机化 |

#### Hydra Overrides（第683行）

通过 `hydra_overrides` 列表进一步调整环境参数:
- `robot=g1/g1_29dof_hard_waist` — 机器人配置
- `robot.control.action_scale=0.25` — 动作缩放
- `robot.control.action_clip_value=5.0` — 动作裁剪
- `robot.control.normalize_action_to=5.0` — 动作归一化范围
- `env.config.lie_down_init=True` — 允许卧倒初始化
- `env.config.lie_down_init_prob=0.3` — 卧倒初始化概率

#### 更多环境参数

位于 Hydra YAML 配置文件 `humanoidverse/config/exp/bfm_zero/bfm_zero.yaml` 及其依赖的子配置中，包含地形、奖励函数、域随机化等详细设置。

### 4.5 WandB 日志参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `use_wandb` | `False` | 是否启用 WandB |
| `wandb_ename` | `'yitangl'` | WandB entity |
| `wandb_gname` | `'bfmzero-isaac'` | 运行组名 |
| `wandb_pname` | `'bfmzero-isaac'` | 项目名 |

### 4.6 评估参数 (第716行)

| 参数 | 值 | 说明 |
|------|-----|------|
| `name` | `HumanoidVerseIsaacTrackingEvaluationConfig` | 评估类型 |
| `num_envs` | `512` | 评估并行环境数 |
| `n_episodes_per_motion` | `1` | 每个动作的评估次数 |
| `generate_videos` | `False` | 是否生成视频 |

---

## 5. 参数修改方式总结

所有参数都在 `train_bfm_zero()` 函数（第587-721行）中以 Python 代码方式硬编码。修改方式：

1. **直接修改 `train.py`**: 编辑 `train_bfm_zero()` 中的 `TrainConfig(...)` 构造参数
2. **Hydra 配置文件**: 环境相关参数通过 `humanoidverse/config/` 下的 YAML 文件配置，并可通过 `hydra_overrides` 列表覆盖
3. **默认值参考**: `humanoidverse/agents/fb_cpr/configs.py` 提供了原始 ICLR 2025 论文的默认配置字典，可作为参数调整的基准参考
