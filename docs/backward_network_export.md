# Backward 网络（B 网络）ONNX 导出说明

本文档说明 BFM-Zero 中 **Backward Map（B 网络）** 的 ONNX 导出方式、输入 / 输出张量约定，以及与训练阶段的对齐细节。

---

## 1. 网络在模型中的位置

- 实现：`humanoidverse/agents/nn_models.py` — `class BackwardMap`
- 配置：`humanoidverse/agents/nn_models.py` — `class BackwardArchiConfig`
- 训练时构造：`humanoidverse/train.py:606`
  ```python
  b = BackwardArchiConfig(
      name='BackwardArchi',
      hidden_dim=256,
      hidden_layers=1,
      norm=True,
      input_filter=DictInputFilterConfig(
          name='DictInputFilterConfig',
          key=['state', 'privileged_state'],
      ),
  )
  ```
- 模型级接口：`humanoidverse/agents/fb/model.py:104`
  ```python
  @torch.no_grad()
  def backward_map(self, obs): ...
  @torch.no_grad()
  def project_z(self, z): ...
  ```
- 在 tracking 推理中的真实用法（`humanoidverse/tracking_inference.py:76`）：
  ```python
  z = model.backward_map(obs)
  z = model.project_z(z)        # √d · F.normalize(z, dim=-1)
  ```

B 网络承担的角色：给定参考轨迹的状态 / 特权状态序列，编码出隐变量 `z`，再被 policy 当作"任务条件"使用。

---

## 2. 网络结构

`BackwardMap.net = Sequential(
    Linear(in_dim, 256), LayerNorm(256), Tanh(),
    Linear(256, 256),   ReLU(),         # 共 hidden_layers - 1 个隐藏层
    ...,
    Linear(256, z_dim),                 # z_dim = 256（来自 train.py:603）
    Norm()                              # cfg.norm=True 时启用
)`

`Norm()` 仅做 `F.normalize(x, dim=-1)`（不带 √d 缩放）；额外的 √d 缩放由 `project_z` 完成。

完整调用链：
```
raw obs dict
   │
   ▼
ObsNormalizer (BatchNorm per key, eval 模式)        # FBModel._normalize
   │
   ▼
DictInputConcatFilter(['state', 'privileged_state'])   # B.input_filter
   │
   ▼
MLP + LayerNorm + Tanh + (ReLU layers) + Linear + Norm
   │
   ▼
project_z(z) = √z_dim · F.normalize(z, dim=-1)
```

---

## 3. 导出函数

### 函数签名

`humanoidverse/utils/helpers.py:291`

```python
def export_backward_map_as_onnx(
    inference_model,                # FBModel / FBcprModel / FBcprAuxModel 实例
    path,                           # 输出目录（不存在会自动创建）
    exported_name,                  # 文件名，例如 "FBcprAuxModel_backward.onnx"
    example_obs=None,               # 可选，dummy 输入；默认随机一份正确 shape 的张量
):
```

### 关键实现细节

1. **维度自适应**：从 `inference_model.obs_space` 读 `state` / `privileged_state` 各自的维度
   ```python
   state_dim = obs_space.spaces["state"].shape[0]
   priv_dim  = obs_space.spaces["privileged_state"].shape[0]
   ```
2. **拷贝到 CPU**：`copy.deepcopy(inference_model).to("cpu")`，避免 GPU 上下文影响导出。
3. **Wrapper 切片重组**：把单一扁平张量切回 `dict` 喂给 `model.backward_map`：
   ```python
   class BWrapper(nn.Module):
       def forward(self, b_obs):
           state = b_obs[:, : self.state_dim]
           priv  = b_obs[:, self.state_dim : self.state_dim + self.priv_dim]
           z = self.model.backward_map({"state": state, "privileged_state": priv})
           return self.model.project_z(z)
   ```
4. **归一化 z**：输出已经过 `project_z`，即 `√z_dim · F.normalize(z)`；部署侧直接当 policy 的 `z` 输入即可，无需再做 `√d · normalize`。
5. **opset**：固定 `opset_version=13`，与 policy 导出保持一致。
6. **缺失键安全**：`ObsNormalizer` 配置了 `allow_mismatching_keys=True`（`train.py:620`），因此只传 `state` / `privileged_state` 不会触发 KeyError。

### 调用入口

在 `humanoidverse/tracking_inference.py:69` 紧跟 policy 导出之后：

```python
export_backward_map_as_onnx(
    model,
    output_dir,                                 # = model_folder / "exported"
    f"{model_name}_backward.onnx",
)
```

跑完一次 tracking 推理后，会在 `<model_folder>/exported/` 下同时产生：
- `<ModelName>_policy.onnx` — actor 网络
- `<ModelName>_backward.onnx` — backward 网络

---

## 4. ONNX 输入 / 输出格式

### 输入

| 名称    | 形状                                      | dtype     | 说明                                                                  |
|---------|-------------------------------------------|-----------|-----------------------------------------------------------------------|
| `b_obs` | `[B, state_dim + privileged_state_dim]`   | `float32` | 沿最后一维拼接 `state` 和 `privileged_state` 两段。`B` 为 batch 维。 |

**切片约定**（与 wrapper 内部一致）：
- `b_obs[:, : state_dim]` → `state`
- `b_obs[:, state_dim : state_dim + privileged_state_dim]` → `privileged_state`

`state` 和 `privileged_state` 的内容定义见 `humanoidverse/utils/helpers.py` 的 `get_backward_observation()`：
- `state = concat([ref_dof_pos, ref_dof_vel, projected_gravity, ref_ang_vel])`，29-DOF 下 `state_dim = 64`
- `privileged_state = max_local_self_obs`，由 `compute_humanoid_observations_max(_with_contact)` 生成（具体 dim 由 robot/env 配置决定，可在运行时通过 `model.obs_space.spaces["privileged_state"].shape[0]` 读取）

### 输出

| 名称 | 形状           | dtype     | 说明                                                                                  |
|------|----------------|-----------|---------------------------------------------------------------------------------------|
| `z`  | `[B, z_dim]`   | `float32` | `project_z(B(obs))`，即 `√z_dim · F.normalize(B(obs), dim=-1)`。`z_dim = 256`（来自训练配置）。 |

输出可直接作为 policy ONNX 输入末尾的 `z` 段使用，**不需要再做归一化**。

---

## 5. 部署侧最小调用示例

```python
import numpy as np
import onnxruntime as ort

sess = ort.InferenceSession("FBcprAuxModel_backward.onnx",
                            providers=["CPUExecutionProvider"])

# 准备一帧或一段参考观测
state = ...              # shape [T, state_dim]
privileged = ...         # shape [T, privileged_state_dim]
b_obs = np.concatenate([state, privileged], axis=-1).astype(np.float32)

z = sess.run(["z"], {"b_obs": b_obs})[0]    # shape [T, z_dim]
# z 已 project_z，可直接拼到 policy 输入末尾使用
```

如果想模拟训练时的 tracking 推理时序平滑（`tracking_inference.py:76`）：
```python
for step in range(z.shape[0]):
    end_idx = min(step + 1, z.shape[0])      # seq_length=1 即只取当前帧
    z[step] = z[step:end_idx].mean(axis=0)
```

---

## 6. 注意事项

- **eval 模式**：导出前会调 `inference_model.eval()`，确保 BatchNorm 走滑动统计、非训练统计。
- **不要在导出后改原模型**：`copy.deepcopy` 后导出独立副本，原模型继续训练 / 推理不受影响。
- **z_dim 与训练一致**：当前 `bfmzero-test` 配置 `z_dim=256`，policy ONNX 末尾的 `z` 段维度必须匹配。
- **observation 归一化**：`obs_normalizer` 的 BatchNorm 参数已经被 `deepcopy` 进 ONNX，**部署时不要再做二次归一化**，直接喂原始 `state` / `privileged_state` 即可。
- **不支持动态 batch（默认）**：`torch.onnx.export` 调用未指定 `dynamic_axes`，导出后 batch 维通常会固定为 dummy 的大小（1）。若部署需要变长 batch，可改成传 `dynamic_axes={"b_obs": {0: "batch"}, "z": {0: "batch"}}`。
