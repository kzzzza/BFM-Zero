# BFM-Zero Deploy

> 在 **Unitree G1（29-DOF）** 上部署 BFM-Zero 策略的完整工程栈，支持 **Sim2Sim（MuJoCo）** 与 **Sim2Real（Jetson Orin）**。控制频率 50 Hz，物理步长 5 ms。

---

## 1. 项目概览

### 1.1 通信架构

仿真器/机器人与策略进程**独立运行**，通过 ZMQ pub-sub 通信；实机时由 `g1_interface`（unitree_sdk2 的 Python 绑定）直接 read/write 机器人。

```
 ┌─────────────────────┐   ZMQ 5590 (LowStateMessage)    ┌─────────────────────┐
 │  Simulator/Robot    │ ──────────────────────────────► │   Policy (50 Hz)    │
 │  - MuJoCo (s2s)     │                                  │  rl_policy/bfm_zero │
 │  - G1Interface(s2r) │ ◄────────────────────────────────│  + ONNX inference   │
 └─────────────────────┘   ZMQ 5591 (LowCmdMessage)      └─────────────────────┘
                          (sim 走 ZMQ；real 走 unitree_sdk2)
```

### 1.2 核心模块

| 路径 | 作用 |
|---|---|
| `rl_policy/bfm_zero.py` | 策略主类，加载 ONNX、组装 obs、50 Hz 调度、发指令 |
| `sim_env/base_sim.py` | MuJoCo 仿真器，物理步长 5 ms，发布 `LowStateMessage` |
| `config/{robot,policy,scene,exp}/` | 机器人 / 策略 / 场景 / 任务的 YAML 配置 |
| `scripts/motion_to_z_onnx.py` | 把 motion `.npz` → 隐变量 z `.pkl`（需要训练之后导出 backward ONNX）|
| `model/` | ONNX 模型 + 三类任务的 z `.pkl`（由 `download_hf_model.py` 拉取） |
| `utils/` | ONNX 包装、四元数数学、Unitree↔Isaac 关节名映射 |
| `deploy_needed/` | 实机预检脚本（ONNX 延迟测试等） |
| `docs/` | sim2real SOP、backward 导出、关节顺序等详细文档 |

---

## 2. 任务模式（配置驱动）

| 模式 | 配置文件 | 隐变量 z 来源 | 切换键 |
|---|---|---|---|
| **tracking** | `config/exp/tracking/walking.yaml` | 预计算 z 序列 `.pkl`，按 `gamma` 折扣窗口加权 | `[` 启动、`p` 复位 |
| **tracking_online** | `config/exp/tracking_online/walking.yaml` | 每帧实时跑 B 网络 ONNX 算 z，支持 `.npz` 文件或 ZMQ 流式输入 | `[` 启动、`p` 复位 |
| **reward** | `config/exp/reward/locomotion.yaml` | 多条 reward 的 z 字典 + `selected_rewards_filter_z` 过滤 | `n` 切换 |
| **goal** | `config/exp/goal/goal.yaml` | `goal_reaching.pkl` 中按名字挑姿态 | `n` 切换 |

> `tracking_online` 的完整使用方法、ZMQ 转发原理、以及如何把自定义动作数据源接入策略，见 [`docs/online_tracking_zmq.md`](docs/online_tracking_zmq.md)。

---

## 3. 快速开始：Sim2Sim（PC运行）

### 3.1 环境准备

```bash
conda create -n bfm0real python=3.10 -y
conda activate bfm0real
cd BFM-Zero_deploy
pip install -r requirements.txt
```

模型和数据准备(放在如下位置，模型从训练结果拷贝，动作数据准备参考下面的第5节)

```
model/
├── exported/FBcprAuxModel.onnx               # 主策略
├── exported/FBcprAuxModel_backward_test.onnx # backward 网络（z 生成）
├── tracking_inference/*.pkl                  # tracking z 序列
├── reward_inference/*.pkl                    # reward z 字典
└── goal_inference/*.pkl                      # goal z 字典
```

### 3.2 启动仿真器（终端 1）

```bash
python -m sim_env.base_sim \
    --robot_config ./config/robot/g1.yaml \
    --scene_config ./config/scene/g1_29dof.yaml
```

> macOS 需用 `mjpython` 替代 `python`。

### 3.3 启动策略（终端 2，三种任务）

```bash
./rl_policy/tracking.sh           # 动作跟踪（预计算 z）
./rl_policy/tracking_online.sh    # 动作跟踪（B 网络在线算 z，支持文件 / ZMQ 流）
./rl_policy/reward.sh             # reward 推理
./rl_policy/goal.sh               # goal 到达
```

`tracking_online` 的详细启动顺序与 ZMQ 接入说明见 [`docs/online_tracking_zmq.md`](docs/online_tracking_zmq.md)。


### 3.4 键盘交互

| 键 | 作用 |
|---|---|
| `i` | 插值到 default 蹲姿（500 步） |
| `]` | 启用策略输出 |
| `[` | tracking：启动动作；`p` 复位 |
| `n` | reward / goal：切换下一个 |
| `o` | 紧急停止（输出归零） |
| `4`/`5` / `6`/`7` / `0` | 调 kp_level（粗调降 / 升 / 复位 1.0） |

---

## 4. Sim2Real（实机部署）


实机流程见 [`docs/sim2real_deployment_sop.md`](docs/sim2real_deployment_sop.md)，

---

## 5.动作转换（z 生成 pipeline）

把 motion 数据 `.npz` 转成隐变量 z `.pkl`，目前不在Jetson上做端侧推导，PC端转换成 `z` 之后做跟踪：

```bash
# tracking 单文件
python scripts/motion_to_z_onnx.py tracking \
    --input  path/to/motion.npz \
    --output model/tracking_inference/my_motion_z.pkl

# tracking 批量（保留目录结构）
python scripts/motion_to_z_onnx.py tracking \
    --input_dir  ./data/motions/ \
    --output_dir model/tracking_inference/

# goal —— 先改 config/z_inference/goal_clips.yaml，再跑
python scripts/motion_to_z_onnx.py goal \
    --clips_config config/z_inference/goal_clips.yaml
```

产物维度：tracking → `ndarray[T, 256]`；goal → `dict[name → ndarray[1, 256]]`，均已 `project_z`，norm = √256 = 16。

生成后修改 `config/exp/<type>/*.yaml` 的 `ctx_path` 指向新 pkl。

backward 网络的 ONNX 导出约定、输入张量切片、归一化细节见 [`docs/backward_network_export.md`](docs/backward_network_export.md)；motion 数据格式与关节顺序见 [`docs/joint_body_ordering.md`](docs/joint_body_ordering.md)。

---

## Citation

```bibtex
@misc{li2025bfmzeropromptablebehavioralfoundation,
      title={BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
      author={Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
      year={2025},
      eprint={2511.04131},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.04131}
}
```

## License

BFM-Zero 采用 CC BY-NC 4.0 协议，详见 [LICENSE](LICENSE)。
