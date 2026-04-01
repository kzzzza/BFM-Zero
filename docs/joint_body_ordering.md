# G1 Joint & Body Ordering Reference

BFM-Zero pipeline 中存在两套关节/body 排列顺序：**Isaac ordering**（模型使用）和 **BMimic ordering**（GMR retarget 输出）。本文档记录两者的完整定义和映射关系。

---

## 1. Isaac Ordering（模型标准顺序）

模型训练、ONNX 推理、`example_motion/` 数据、deploy 策略均使用此顺序。来源：MuJoCo XML 中的关节定义顺序（跳过 floating_base_joint）。

### 1.1 Joint Ordering（29-DOF）

按肢体分组：左腿 → 右腿 → 腰 → 左臂 → 右臂

| Isaac Index | Joint Name |
|:-----------:|:-----------|
| 0  | left_hip_pitch_joint |
| 1  | left_hip_roll_joint |
| 2  | left_hip_yaw_joint |
| 3  | left_knee_joint |
| 4  | left_ankle_pitch_joint |
| 5  | left_ankle_roll_joint |
| 6  | right_hip_pitch_joint |
| 7  | right_hip_roll_joint |
| 8  | right_hip_yaw_joint |
| 9  | right_knee_joint |
| 10 | right_ankle_pitch_joint |
| 11 | right_ankle_roll_joint |
| 12 | waist_yaw_joint |
| 13 | waist_roll_joint |
| 14 | waist_pitch_joint |
| 15 | left_shoulder_pitch_joint |
| 16 | left_shoulder_roll_joint |
| 17 | left_shoulder_yaw_joint |
| 18 | left_elbow_joint |
| 19 | left_wrist_roll_joint |
| 20 | left_wrist_pitch_joint |
| 21 | left_wrist_yaw_joint |
| 22 | right_shoulder_pitch_joint |
| 23 | right_shoulder_roll_joint |
| 24 | right_shoulder_yaw_joint |
| 25 | right_elbow_joint |
| 26 | right_wrist_roll_joint |
| 27 | right_wrist_pitch_joint |
| 28 | right_wrist_yaw_joint |

### 1.2 Body Ordering（31 bodies，含 head_link）

来源：env.py `get_privileged_state()` 的选取逻辑（排除 world、dummy_*、*_hand，head_link 放末尾）。

| Isaac Index | Body Name |
|:-----------:|:----------|
| 0  | pelvis |
| 1  | left_hip_pitch_link |
| 2  | left_hip_roll_link |
| 3  | left_hip_yaw_link |
| 4  | left_knee_link |
| 5  | left_ankle_pitch_link |
| 6  | left_ankle_roll_link |
| 7  | right_hip_pitch_link |
| 8  | right_hip_roll_link |
| 9  | right_hip_yaw_link |
| 10 | right_knee_link |
| 11 | right_ankle_pitch_link |
| 12 | right_ankle_roll_link |
| 13 | waist_yaw_link |
| 14 | waist_roll_link |
| 15 | torso_link |
| 16 | left_shoulder_pitch_link |
| 17 | left_shoulder_roll_link |
| 18 | left_shoulder_yaw_link |
| 19 | left_elbow_link |
| 20 | left_wrist_roll_link |
| 21 | left_wrist_pitch_link |
| 22 | left_wrist_yaw_link |
| 23 | right_shoulder_pitch_link |
| 24 | right_shoulder_roll_link |
| 25 | right_shoulder_yaw_link |
| 26 | right_elbow_link |
| 27 | right_wrist_roll_link |
| 28 | right_wrist_pitch_link |
| 29 | right_wrist_yaw_link |
| 30 | head_link |

> `example_motion/` 格式只有 30 bodies（不含 head_link），对应 Isaac body index 0–29。

---

## 2. BMimic Ordering（GMR retarget 输出）

`~/repos/GMR` 中 retarget 脚本输出的 `data/` 格式（amass/lafan）使用此顺序。按关节层级分组：所有 pitch → 所有 roll → 所有 yaw → ...

### 2.1 Joint Ordering（29-DOF）

| BMimic Index | Joint Name | Isaac Index |
|:------------:|:-----------|:-----------:|
| 0  | left_hip_pitch_joint       | 0  |
| 1  | right_hip_pitch_joint      | 6  |
| 2  | waist_yaw_joint            | 12 |
| 3  | left_hip_roll_joint        | 1  |
| 4  | right_hip_roll_joint       | 7  |
| 5  | waist_roll_joint           | 13 |
| 6  | left_hip_yaw_joint         | 2  |
| 7  | right_hip_yaw_joint        | 8  |
| 8  | waist_pitch_joint          | 14 |
| 9  | left_knee_joint            | 3  |
| 10 | right_knee_joint           | 9  |
| 11 | left_shoulder_pitch_joint  | 15 |
| 12 | right_shoulder_pitch_joint | 22 |
| 13 | left_ankle_pitch_joint     | 4  |
| 14 | right_ankle_pitch_joint    | 10 |
| 15 | left_shoulder_roll_joint   | 16 |
| 16 | right_shoulder_roll_joint  | 23 |
| 17 | left_ankle_roll_joint      | 5  |
| 18 | right_ankle_roll_joint     | 11 |
| 19 | left_shoulder_yaw_joint    | 17 |
| 20 | right_shoulder_yaw_joint   | 24 |
| 21 | left_elbow_joint           | 18 |
| 22 | right_elbow_joint          | 25 |
| 23 | left_wrist_roll_joint      | 19 |
| 24 | right_wrist_roll_joint     | 26 |
| 25 | left_wrist_pitch_joint     | 20 |
| 26 | right_wrist_pitch_joint    | 27 |
| 27 | left_wrist_yaw_joint       | 21 |
| 28 | right_wrist_yaw_joint      | 28 |

### 2.2 Body Ordering（30 bodies，不含 head_link）

| BMimic Index | Body Name | Isaac Index |
|:------------:|:----------|:-----------:|
| 0  | pelvis                    | 0  |
| 1  | left_hip_pitch_link       | 1  |
| 2  | right_hip_pitch_link      | 7  |
| 3  | waist_yaw_link            | 13 |
| 4  | left_hip_roll_link        | 2  |
| 5  | right_hip_roll_link       | 8  |
| 6  | waist_roll_link           | 14 |
| 7  | left_hip_yaw_link         | 3  |
| 8  | right_hip_yaw_link        | 9  |
| 9  | torso_link                | 15 |
| 10 | left_knee_link            | 4  |
| 11 | right_knee_link           | 10 |
| 12 | left_shoulder_pitch_link  | 16 |
| 13 | right_shoulder_pitch_link | 23 |
| 14 | left_ankle_pitch_link     | 5  |
| 15 | right_ankle_pitch_link    | 11 |
| 16 | left_shoulder_roll_link   | 17 |
| 17 | right_shoulder_roll_link  | 24 |
| 18 | left_ankle_roll_link      | 6  |
| 19 | right_ankle_roll_link     | 12 |
| 20 | left_shoulder_yaw_link    | 18 |
| 21 | right_shoulder_yaw_link   | 25 |
| 22 | left_elbow_link           | 19 |
| 23 | right_elbow_link          | 26 |
| 24 | left_wrist_roll_link      | 20 |
| 25 | right_wrist_roll_link     | 27 |
| 26 | left_wrist_pitch_link     | 21 |
| 27 | right_wrist_pitch_link    | 28 |
| 28 | left_wrist_yaw_link       | 22 |
| 29 | right_wrist_yaw_link      | 29 |

---

## 3. 转换映射（Python）

### 3.1 Joint 映射

```python
# Isaac → BMimic（即 GMR 中的 G1_JOINT_MAPPING）
# 含义：bmimic_joints[i] = isaac_joints[ISAAC_TO_BMIMIC[i]]
ISAAC_TO_BMIMIC = [
    0, 6, 12,
    1, 7, 13,
    2, 8, 14,
    3,  9, 15, 22,
    4, 10, 16, 23,
    5, 11, 17, 24,
          18, 25,
          19, 26,
          20, 27,
          21, 28,
]

# BMimic → Isaac（逆映射）
# 含义：isaac_joints[j] = bmimic_joints[BMIMIC_TO_ISAAC[j]]
BMIMIC_TO_ISAAC = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
```

使用示例：

```python
import numpy as np

# data/ (BMimic) → Isaac
joint_pos_isaac = joint_pos_bmimic[:, BMIMIC_TO_ISAAC]

# Isaac → data/ (BMimic)
joint_pos_bmimic = joint_pos_isaac[:, ISAAC_TO_BMIMIC]
```

### 3.2 Body 映射

```python
# BMimic(30) → Isaac(30, 不含 head_link)
BMIMIC_TO_ISAAC_BODY = [0, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10, 16, 23, 5, 11, 17, 24, 6, 12, 18, 25, 19, 26, 20, 27, 21, 28, 22, 29]
```

---

## 4. NPZ 数据格式

### 4.1 `example_motion/` 格式（Isaac ordering）

| Key | Shape | Description |
|:----|:------|:------------|
| `dof_positions` | `[T, 29]` | 关节角度（rad），Isaac ordering |
| `body_positions` | `[T, 30, 3]` | Body 位置（world frame），Isaac body ordering（无 head_link） |
| `body_rotations` | `[T, 30, 4]` | Body 四元数 `[w, x, y, z]`，Isaac body ordering |
| `dof_names` | `[29]` | 关节名列表（Isaac ordering） |
| `fps` | scalar | 帧率（通常 50） |

无速度数据，需通过有限差分计算。

### 4.2 `data/` 格式（BMimic ordering）

| Key | Shape | Description |
|:----|:------|:------------|
| `joint_pos` | `[T, 29]` | 关节角度（rad），**BMimic ordering** |
| `joint_vel` | `[T, 29]` | 关节角速度（rad/s），**BMimic ordering** |
| `body_pos_w` | `[T, 30, 3]` | Body 位置（world frame），BMimic body ordering |
| `body_quat_w` | `[T, 30, 4]` | Body 四元数 `[w, x, y, z]`，BMimic body ordering |
| `body_lin_vel_w` | `[T, 30, 3]` | Body 线速度（world frame），BMimic body ordering |
| `body_ang_vel_w` | `[T, 30, 3]` | Body 角速度（world frame），BMimic body ordering |
| `fps` | `[1]` int64 | 帧率（通常 50） |

由 `~/repos/GMR` retarget 脚本生成，使用前需转换到 Isaac ordering。

---

## 5. 其他约定

- **四元数格式**：全局统一 `[w, x, y, z]`
- **角度单位**：全局统一 rad
- **坐标系**：world frame，z-up
- **帧率**：通常 50 Hz，与 deploy 控制频率一致
