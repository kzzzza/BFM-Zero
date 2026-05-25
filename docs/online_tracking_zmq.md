# 在线 Tracking 推理（B 网络实时 z 计算）

本文档说明 `tracking_online` 任务的使用方法、ZMQ 转发原理，以及如何把其他动作数据源接入 ZMQ 接口。

与默认 `tracking` 模式（加载预计算的 `.pkl`）不同，`tracking_online` 在每个策略步用 **backward 网络（B 网络）** 实时把当前动作帧编码成 256 维隐变量 z，再喂给策略网络。两种 motion 来源：

- **file**：启动时一次性加载 `.npz`，按 `self.t` 索引（与离线 pkl 行为数值等价）
- **zmq**：外部进程通过 ZMQ 实时推送 `MotionFrameMessage`

---

## 1. 快速对比：tracking vs tracking_online

| 维度 | `tracking`（默认） | `tracking_online` |
|---|---|---|
| z 来源 | 启动时 `joblib.load(.pkl)` | 每帧实时 B 网络推理 |
| 切片粒度 | 整段动作一次性算完 | 每步 1 次 backward ONNX |
| 是否需要预计算 | 是（`scripts/motion_to_z_onnx.py`） | 否 |
| 是否支持流式输入 | 否 | 是（ZMQ 模式） |
| 滑窗平均 | gamma 折扣 + window | forward-looking mean + project（file 模式） |
| 稳态步耗时 | ~2 ms | ~3–5 ms |

源代码索引：
- 主类分支：`rl_policy/bfm_zero.py:_setup_tracking_online` / `_compute_online_z`
- z 计算封装：`rl_policy/utils/online_z_provider.py:OnlineZProvider`
- ZMQ 订阅：`rl_policy/utils/motion_subscriber.py:MotionSubscriber`
- 消息格式：`utils/common.py:MotionFrameMessage`
- 示例生产者：`scripts/replay_motion_zmq.py`
- 配置：`config/exp/tracking_online/walking.yaml`
- 等价性验证：`scripts/test_online_z_equivalence.py`

---

## 2. 使用方法

### 2.1 file 模式（最简单，离线复现）

把 `config/exp/tracking_online/walking.yaml` 改成：

```yaml
type: tracking_online
source: file
seq_length: 1
scene_xml: data/robots/g1/g1_for_backward_obs.xml
backward_onnx: model/exported/FBcprAuxModel_backward_test.onnx
motion_path: data/motions/amass/Form_1_stageii.npz
start: 0
end: 5000
stop: 0
```

两个终端：

```bash
# 终端 1
python -m sim_env.base_sim --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof.yaml

# 终端 2
./rl_policy/tracking_online.sh
```

终端 2 中：按 `i` 初始化姿态 → 按 `]` 启用策略 → 按 `[` 启动动作。

`seq_length>1` 会启用前瞻滑窗平均，与 `scripts/motion_to_z_onnx.py` 离线产物**逐帧 bitwise 等价**（验证脚本见 §6）。

### 2.2 zmq 模式（流式接入）

把 `config/exp/tracking_online/walking.yaml` 改成：

```yaml
type: tracking_online
source: zmq
seq_length: 1                # zmq 模式强制为 1（无法前瞻）
scene_xml: data/robots/g1/g1_for_backward_obs.xml
backward_onnx: model/exported/FBcprAuxModel_backward_test.onnx
zmq_port: 5592
zmq_ip: localhost
```

**严格按 3 个终端的顺序启动**（顺序错会丢前几十帧 —— 见 §3）：

```bash
# 终端 1：仿真器
python -m sim_env.base_sim --robot_config ./config/robot/g1.yaml --scene_config ./config/scene/g1_29dof.yaml

# 终端 2：策略（动作订阅，必须先就位）
./rl_policy/tracking_online.sh
# 等待打印：
#   MotionSubscriber listening on tcp://localhost:5592
#   tracking_online: backward ONNX warmed up
#   Policy ONNX warmed up (...)

# 终端 3：动作传输（最后启动）
python scripts/replay_motion_zmq.py \
    --motion data/motions/amass/Form_1_stageii.npz \
    --rate 50
```

回到终端 2：按 `i` → `]` → `[`。

`replay_motion_zmq.py` 选项：

| 选项 | 说明 |
|---|---|
| `--motion <path>` | 必填，支持 `data/` 与 `example_motion/` 两种 `.npz` 格式 |
| `--rate <Hz>` | 推送频率，默认 50（与策略一致） |
| `--start <int> --end <int>` | 切片范围，`--end -1` 表示整段 |
| `--loop` | 循环播放（不发 END flag） |
| `--port <int>` | 端口，必须与 `walking.yaml` 的 `zmq_port` 一致 |

### 2.3 键盘交互（两种模式通用）

| 键 | 作用 |
|---|---|
| `i` | 插值到 default 蹲姿（500 步） |
| `]` | 启用策略输出 |
| `[` | 启动 tracking_online：重置 z 缓存 + （file 模式）预填 seq_length 帧 + 翻 `start_motion=True` |
| `p` | 停止 motion，保持当前姿态 |
| `o` | 紧急停止（输出归零） |

摇杆映射（USE_JOYSTICK=True 时）：`A` ≈ `i`、`R1` ≈ `]`、`B` ≈ `[`、`X` ≈ `p`、`R2` ≈ `o`。

---

## 3. ZMQ 转发原理

### 3.1 通信链路

```
 ┌──────────────────────┐   tcp://*:5592 (PUB)         ┌──────────────────────────┐
 │  Motion Producer     │ ───────────────────────────► │  MotionSubscriber (SUB)  │
 │  (你的数据源 / 重放) │   MotionFrameMessage (292B)   │  daemon thread → deque   │
 └──────────────────────┘                              └──────────────────────────┘
                                                                   │
                                                                   │ poll()
                                                                   ▼
                                                       ┌──────────────────────────┐
                                                       │  OnlineZProvider          │
                                                       │  set_state → mj_forward  │
                                                       │  → backward ONNX → z      │
                                                       └──────────────────────────┘
                                                                   │
                                                                   ▼
                                                       策略观测拼接 → policy ONNX → 关节命令
```

### 3.2 关键设计点

1. **PUB-SUB 单向广播**：生产者 `bind`，订阅者 `connect`。PUB 在没有订阅者连接时**直接丢消息**，所以**必须先起订阅者**。
2. **CONFLATE=0**：`MotionSubscriber` 关闭了 conflation。常用的 `utils.common.ZMQSubscriber` 默认 `CONFLATE=1` 会保留最新一条丢弃中间，**绝对不能这样**——`BackwardObsBuilder` 的 body 速度是相邻两帧的有限差分，跳帧 = z 计算错误。
3. **bounded deque（maxlen=64）**：daemon 线程阻塞收 → 立即解码 → append 到 deque。策略线程 `poll()` 非阻塞 popleft。背压自然丢最老的（如果策略掉队）。
4. **不传 body 数组**：消息只含根 + 关节状态。body 位置 / 朝向由 `mj_forward` 在订阅端重建 —— 保证与离线路径 kinematics 完全一致，也省带宽（292B/帧 vs ~2KB/帧）。
5. **`frame_idx` 驱动 `self.t`**：策略循环用消息里的 `frame_idx` 作为当前帧索引。生产者跳号 / 回退 / 重启 → `OnlineZProvider` 自动 `reset()` + warning。
6. **下溢复用 z**：若策略线程一个 step 周期内没收到新帧，复用上一次的 z（debug 日志），保持策略响应。

### 3.3 频率与时延约束

- 策略循环固定 50 Hz（20 ms 预算）。
- 生产者应该**至少**按 50 Hz 推送；快一点没关系（queue 缓冲），慢了会出现 underrun（z 重复一帧）。
- 内部计算（mj_forward + backward ONNX）稳态 ~3 ms / 帧，留有充足余量。
- 首次 ONNX 推理有冷启动成本，已在 `_setup_tracking_online` 和 `setup_policy` 末尾通过 dummy warmup 摊销。

---

## 4. ZMQ 接口：接入自定义动作数据源

如果你要把别的来源（实时动捕、上位机轨迹规划、网络遥操作、其他 motion 数据库等）接入策略，只需要做一个 PUB 端发 `MotionFrameMessage` 到端口 5592 即可。

### 4.1 `MotionFrameMessage` 字段定义

定义在 `utils/common.py:MotionFrameMessage`。每条消息共 292 字节：

| 字段 | 类型 / 维度 | 说明 |
|---|---|---|
| `frame_idx` | uint32 | 单调递增的帧索引；策略用它做 `self.t` |
| `flags` | uint32 | bit0 = `FLAG_END`（动作结束，策略收到后停 motion）；其余位保留 |
| `joint_pos` | float32 × 29 | 关节角，**Isaac 顺序**（见 `docs/joint_body_ordering.md`） |
| `joint_vel` | float32 × 29 | 关节速度，Isaac 顺序 |
| `root_pos` | float32 × 3 | 根 (pelvis) 在世界系下的位置 |
| `root_quat` | float32 × 4 | 根四元数 `[w, x, y, z]`，世界系 |
| `root_lin_vel_w` | float32 × 3 | 根线速度，**世界系** |
| `root_ang_vel_w` | float32 × 3 | 根角速度，**世界系** |

> 关于 Isaac 关节顺序：参见 `docs/joint_body_ordering.md`。如果你的数据是 BMimic / GMR 顺序，参考 `scripts/utils/motion_loader.py:G1_JOINT_MAPPING` 做重排。

### 4.2 二进制布局

```
+--------+--------+----------------------+
|frame_  | flags  | 71 × float32 payload |
| idx    |        |                      |
| u32 LE | u32 LE | 284 bytes            |
+--------+--------+----------------------+
   4 B      4 B             284 B            = 292 B 总长
```

`payload` 按以下顺序拼接（与字段表一致）：
`joint_pos(29) | joint_vel(29) | root_pos(3) | root_quat(4) | root_lin_vel_w(3) | root_ang_vel_w(3)`

### 4.3 Python 最小生产者示例

```python
import time
import numpy as np
import zmq

import sys
sys.path.insert(0, "/path/to/BFM-Zero_deploy")
from utils.common import PORTS, MotionFrameMessage

ctx = zmq.Context.instance()
sock = ctx.socket(zmq.PUB)
sock.bind(f"tcp://*:{PORTS['motion_frame']}")  # 5592
time.sleep(0.5)  # 给订阅者握手时间

frame_idx = 0
rate_hz = 50
dt = 1.0 / rate_hz
next_t = time.perf_counter()

while True:
    # === 在这里替换成你的真实数据源 ===
    joint_pos      = np.zeros(29, dtype=np.float32)
    joint_vel      = np.zeros(29, dtype=np.float32)
    root_pos       = np.array([0.0, 0.0, 0.75], dtype=np.float32)
    root_quat      = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # [w,x,y,z]
    root_lin_vel_w = np.zeros(3, dtype=np.float32)
    root_ang_vel_w = np.zeros(3, dtype=np.float32)
    is_last        = False
    # ================================

    msg = MotionFrameMessage(
        frame_idx=frame_idx,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        root_pos=root_pos,
        root_quat=root_quat,
        root_lin_vel_w=root_lin_vel_w,
        root_ang_vel_w=root_ang_vel_w,
        flags=MotionFrameMessage.FLAG_END if is_last else 0,
    )
    sock.send(msg.to_bytes())

    frame_idx += 1
    next_t += dt
    sleep_for = next_t - time.perf_counter()
    if sleep_for > 0:
        time.sleep(sleep_for)
    else:
        next_t = time.perf_counter()  # 落后时重置调度避免突发追赶
```

完整的可执行参考实现见 `scripts/replay_motion_zmq.py`。

### 4.4 非 Python 端接入

只要按 §4.2 的字节布局发送，任何语言都可以。建议步骤：

1. 创建 `zmq.PUB` socket，`bind` 到 `tcp://*:5592`。
2. 拼装 292 字节：`<frame_idx u32_LE><flags u32_LE><71 个 float32_LE>`，按字段顺序填充。
3. `socket.send(bytes)` 一次发一条 message。
4. 按目标频率（建议 50 Hz）循环。
5. 动作结束时把 `flags |= 0x1` 标记 END，之后 socket close。

C++ / Rust / Go 端可以参考 ZMQ 官方文档的 PUB socket 示例，结合上面的二进制布局完成实现。

### 4.5 端到端调试链路（不依赖仿真器）

调试自己的生产者时，先用一个独立的 dump 订阅者验证字节布局对：

```bash
python -c "
import zmq, sys
sys.path.insert(0, '.')
from utils.common import PORTS, MotionFrameMessage
ctx = zmq.Context.instance()
s = ctx.socket(zmq.SUB)
s.connect(f'tcp://localhost:{PORTS[\"motion_frame\"]}')
s.setsockopt(zmq.SUBSCRIBE, b'')
for _ in range(5):
    m = MotionFrameMessage.from_bytes(s.recv())
    print(f'frame_idx={m.frame_idx} flags={m.flags} '
          f'root_pos={m.root_pos} root_quat={m.root_quat} '
          f'joint_pos[:3]={m.joint_pos[:3]}')
"
```

如果能连续打印 5 帧且数值合理，说明你的生产者格式正确。然后再换成完整的策略 + 仿真链路。

---

## 5. 注意事项与常见坑

| 现象 | 原因 / 处理 |
|---|---|
| 策略端日志 `OnlineZProvider: discontinuity (last=X, new=Y); auto-resetting` | 生产者跳帧、回退、或重启。auto-reset 后首帧的 body 速度会被置 0，单帧扰动一般可接受。如果是稳态频发，检查生产者 `frame_idx` 是否每次 `+1` |
| 策略动作没反应 | 没按 `]`（策略未启用）或 `[`（motion 未启动） |
| 策略收到的首帧 `frame_idx` 是 200+（不是 0） | 生产者先于订阅者起动，前面帧全丢。严格按 §2.2 顺序启动 |
| `RL step took 0.020XXX seconds` 偶发 | 通常是 ONNX 冷启动 / 偶发抖动，warmup 之后应消失。持续超时再排查 `self.perf_dict` |
| 策略端持续 `underrun, reusing last z`（debug 级） | 生产者频率不足或已停。检查终端 3 状态 |
| 关节顺序不对 / 动作乱抖 | 数据没按 Isaac 关节顺序排，见 `docs/joint_body_ordering.md` |
| 四元数符号反转后机器人姿态错 | `root_quat` 必须是 `[w, x, y, z]` 顺序，不是 `[x, y, z, w]` |
| 实时数据没有 `joint_vel` / `root_lin_vel_w` 等导数项 | 在生产者端用前向差分自行计算（`scripts/utils/motion_loader.py:_compute_velocities_finite_diff` 是参考实现） |

---

