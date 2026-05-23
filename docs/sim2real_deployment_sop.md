# BFM-Zero Sim2Real 真机部署 SOP

> **适用范围**：已通过 ssh 连接到 Unitree G1 板载 Jetson Orin，准备从零完成第一次真机部署直到跑通策略。
>
> **前置 sim2sim**：本文档假设你已经在本地 PC 上跑通了 sim2sim（MuJoCo），且当前要部署的 task yaml + z pkl 已经在仿真里验证过姿态/步态正常。
> 
> **安全红线**：跑策略前**必须挂吊带或 elastic band 起吊**；R2（手柄）/ `o`（键盘）的位置必须在试机前用手指找到。

---

## 约定

- `[本地]` 前缀：在本地开发 PC 上执行
- `[Jetson]` 前缀：通过 ssh 在 Jetson Orin 上执行
- `[手柄]` / `[物理]` 前缀：机器人物理操作
- 用 `<尖括号>` 包起来的是占位符，替换成你自己的值
- 假设机器人 IP 为 `192.168.123.164`，用户名 `unitree`

---

## Phase 0：连接 & 代码移植（5 分钟）

### 0.1 ssh 连接

1. 使用网线/局域网链接机器人，注意需要打开网络设置，修改电脑和机器人网络**处在统一子网网段**

2. ssh链接到机器人：
   
```bash
# [本地]
ssh unitree@192.168.123.164
# 机器人默认密码：123
```

3. 进入 Jetson 后，先确认环境基本信息：

```bash
# [Jetson]
uname -a                         # 应显示 aarch64 (Linux 5.x ... aarch64 GNU/Linux)
ip a                             # 记下与机器人通信的网卡名（通常 eth0）
df -h ~                          # 确认 home 还有 >5GB 可用
```


### 0.2 拷贝仓库到 Jetson（仅首次）

由于本地的代码仓库会有一些不需要移植到端侧的目录，推荐按照以下途径移植
   
```bash
# [本地] 先加上参数 n 模拟传输，查看会发送哪些文件，总数大概在150MB左右 
rsync -avhn --exclude='.venv' --exclude='venv' \
      --exclude='.git' --exclude='.claude' --exclude='.vscode' --exclude='.idea' \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
      --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
      --exclude='*.egg-info' --exclude='*.log' --exclude='.DS_Store' \
      --exclude='model' --exclude='video' --exclude='results' \
      /home/<使用PC上仓库的路径>/BFM-Zero_deploy/ \
      unitree@192.168.123.164:/home/unitree/workspace/<修改为你自己的工作目录>/BFM-Zero_deploy/   

# [本地] 确认无误之后，删除n参数重新传输脚本文件
rsync -avhn --exclude='.venv' --exclude='venv' \
      --exclude='.git' --exclude='.claude' --exclude='.vscode' --exclude='.idea' \
      --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pyo' \
      --exclude='.pytest_cache' --exclude='.mypy_cache' --exclude='.ruff_cache' \
      --exclude='*.egg-info' --exclude='*.log' --exclude='.DS_Store' \
      --exclude='model' --exclude='video' --exclude='results' \
      /home/<使用PC上仓库的路径>/BFM-Zero_deploy/ \
      unitree@192.168.123.164:/home/unitree/workspace/<修改为你自己的工作目录>/BFM-Zero_deploy/ 

# [本地] 模型和动作数据单独传输
rsync -avhP /home/<使用PC上仓库的路径>/BFM-Zero_deploy/model/  unitree@192.168.123.164:/home/unitree/workspace/<修改为你自己的工作目录>/BFM-Zero_deploy/model/ 

```

或者也可以使用 git clone


## Phase 1：Python 环境（首次部署约 10 分钟）

### 1.1 安装 / 激活 conda 环境

```bash
# [Jetson]
# 如果还没装 miniforge3（Jetson 推荐 miniforge 而非 anaconda）：
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3
# source ~/miniforge3/etc/profile.d/conda.sh

conda create -n bfm0real python=3.10 -y
conda activate bfm0real
```

### 1.2 安装依赖

```bash
# [Jetson]
cd ~/workspace/<修改为你自己的工作目录>/BFM-Zero_deploy
pip install -r requirements.txt
# 如果国内安装依赖速度满，推荐使用清华源安装
# pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

```

### 1.3 验证 ONNX GPU 可用

通常需要根据实际Jetson版本重新安装正确的 onnxruntime

```bash
# [Jetson]
python -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望输出：['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# 如果没有输出 'TensorrtExecutionProvider', 'CUDAExecutionProvider'，就需要重新安装
```

**如果只有 CPUExecutionProvider**，实机推理会到 30+ ms，50 Hz 跟不上，必须先解决 GPU provider 问题：

1. 卸载之前的 onnxruntime
   1. ` pip list | grep onnx*` 查看有哪些onnxruntime相关包
   2. ` pip uninstall -y onnxruntime ` 卸载
2. 确认Jetson版本：`cat /etc/nv_tegra_release`
在实验室当前G1输出为`# R35 (release), REVISION: 3.1, GCID: 32827747, BOARD: t186ref, EABI: aarch64, DATE: Sun Mar 19 15:19:21 UTC 2023`
3. 在 [ONNXRuntime](https://elinux.org/Jetson_Zoo#ONNX_Runtime)下载你的Jerson版本对应的 onnxruntime
4. 使用 `BFM-Zero_deploy/deploy_needed/fix_whl_name.py` 处理下载好的whl文件
> 由于网页下载的whl命名错误，不满足pip install安装需要的命名格式，需要对whl文件改名
> 直接运行 python `fix_whl_name.py <你下载的whl文件名>`
> 会自动改名
5. 改好的whl传到端侧，在正确的conda环境运行 `pip install <改名之后的 whl文件名>`
6. 安装好之后，运行`BFM-Zero_deploy/deploy_needed/test_onnx_latency.py`测试端侧ONNXRuntime能否正常工作，通常输出的延迟应该在5ms以下。

---

## Phase 2：编译 Unitree SDK（首次部署约 15 分钟）

### 2.1 安装系统依赖

```bash
# [Jetson]
conda activate bfm0real
sudo apt-get update
sudo apt-get install -y build-essential cmake python3-dev python3-pip git
pip install pybind11 pybind11-stubgen
```

### 2.2 编译 CycloneDDS


```bash
# [Jetson]
# 如果端侧 git clone下载太慢，可以在PC下载好后scp传输
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir -p build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install -j$(nproc)
```

写入环境变量并立刻生效：

```bash
# [Jetson]
echo "export CYCLONEDDS_HOME=<实际安装位置>/cyclonedds/install" >> ~/.bashrc
source ~/.bashrc
echo $CYCLONEDDS_HOME      # 验证：应打印路径
```

### 2.3 编译 unitree_sdk2（EGalahad fork，带 g1_interface 绑定）

```bash
# [Jetson]
# 如果端侧 git clone下载太慢，可以在PC下载好后scp传输
git clone https://github.com/EGalahad/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build

conda activate bfm0real
PYBIND11_DIR=$(python3 -m pybind11 --cmakedir)
echo "pybind11 cmake dir: $PYBIND11_DIR"   # 验证路径非空

cmake .. -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=$PYBIND11_DIR 2>&1 | tee cmake.log
grep -i "Found pybind11" cmake.log         # 必须命中，否则 Python 绑定不会编

make -j$(nproc) 2>&1 | tee make.log
grep -i "Built target g1_interface" make.log   # 必须命中
```

### 2.4 验证 `g1_interface.so` 存在

```bash
# [Jetson]
ls <实际安装位置>/unitree_sdk2/build/lib/
# 期望看到类似：g1_interface.cpython-310-aarch64-linux-gnu.so

# 测试能否 import
python -c "import sys; sys.path.append('$HOME/unitree_sdk2/build/lib'); import g1_interface; print(g1_interface.__file__)"
```

**如果 import 失败**：
- `libddsc.so.0` not found → `CYCLONEDDS_HOME` 没生效，重新 `source ~/.bashrc`
- 找不到 g1_interface → `build/lib` 目录或文件名不对，重看 2.3 的 grep 输出
cd ，可能是运行 `PYBIND11_DIR=$(python3 -m pybind11 --cmakedir)` 时候没有使用正确的conda环境，导致使用错误的路径
---

## Phase 3：代码 & 模型部署

### 3.1 改 `bfm_zero.py` 的 SDK 路径

修改 `rl_policy/bfm_zero.py` 42行的路径为端侧unitree_sdk2的lib实际路径

```bash
# [Jetson]
# 验证替换成功
grep -n "sys.path.append" rl_policy/bfm_zero.py | head -3
``` 

### 3.2 配置 `g1_real.yaml`

确认网卡名：

```bash
# [Jetson]
ip -o link show | awk -F': ' '{print $2}' | grep -v "lo\|docker"
# 输出示例：eth0
```

编辑配置

```bash
# [Jetson]
vim ~/BFM-Zero_deploy/config/robot/g1_real.yaml
```

确认前几行：

```yaml
ROBOT_TYPE: 'g1_real'
DOMAIN_ID: 0
INTERFACE: "eth0"          # 改成 0.3 里查到的实际网卡名
USE_JOYSTICK: True         # 开启手柄控制
```

### 3.3 选定任务 yaml

以 tracking 为例：

```bash
# [Jetson]
cat config/exp/tracking/walking.yaml
```
确认里面 `ctx_path` 指向的 pkl 是你实际要测试的动作（已经映射为 z 向量格式的）。

---

## Phase 4：启动策略

### 4.1 用 tmux 启动（推荐）

```bash
# [Jetson]
tmux new -s bfm
conda activate bfm0real
cd <实际路径>/BFM-Zero_deploy
```

### 4.2 运行控制脚本

`./rl_policy/tracking_real.sh`

或者运行：
```bash
# [Jetson]
python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1_real.yaml \
    --policy_config config/policy/motivo_newG1.yaml \
    --model_path ./model/exported/FBcprAuxModel.onnx \
    --task config/exp/tracking/walking.yaml
```

切换不同 task（reward / goal），只需改 `--task` 参数；

切换不同 motion，只需改 `config/exp/tracking/*.yaml` 里的 `ctx_path`。

### 4.3 启用流程（手柄序列）

> ⚠️ **`USE_JOYSTICK: True` 时键盘 listener 不会启动**，所有控制必须用手柄。键盘的 `4/5/6/7/0` 在 s2r 模式下**完全没有效果**——调 kp_level 必须用下表的 L1/L2 combo。

**完整手柄按键映射**（含 L1/L2 modifier combo）：

| 手柄 | 作用 | 键盘等价（仅 s2s） |
|---|---|---|
| `R1` | 启用策略 / tracking 切下一段 | `]` |
| `R2` | **紧急归零**（输出全部清零） | `o` |
| `A` | 回 default 蹲姿 | `i` |
| `B` | tracking：启动 z 序列 | `[` |
| `X` | 复位 z（停在 stop 帧） | `p` |
| `Y` | reward / goal：切换下一个 | `n` |
| `L1 + Y` / `L1 + A` / `L1 + B` | kp_level **+0.1** / **−0.1** / **= 1.0** | `7` / `4` / `0` |
| `L2 + Y` / `L2 + A`             | kp_level **+0.01** / **−0.01** | `6` / `5` |


---

## 故障排查速查表

| 现象 | 可能原因 | 排查命令 |
|---|---|---|
| `ModuleNotFoundError: g1_interface` | `sys.path.append` 路径错或没编 | `ls <实际安装路径>/unitree_sdk2/build/lib/g1_interface*` |
| `libddsc.so.0: cannot open` | `CYCLONEDDS_HOME` 没生效 | `echo $CYCLONEDDS_HOME`，`source ~/.bashrc` |
| `G1Interface("eth0")` 卡住 | 网卡名错 / Domain ID 不一致 / 机器人未解锁 | `ip a`，检查 `g1_real.yaml` 的 `INTERFACE` 和 `DOMAIN_ID` |
| `RL step took 0.04 expected 0.02` | 50 Hz 跟不上 | `sudo nvpmodel -m 0; sudo jetson_clocks`；关其他进程 |
| ONNX 推理 >15 ms | onnxruntime 在 CPU 跑 | `import onnxruntime; print(ort.get_available_providers())` |
| 步态不对、抖动 | 机器人配置不对（使用G1 Type5） | 换机器人 OR 重新训练|
| `Available z_dict` 为空 | `ctx_path` 相对路径解析错（相对 `model_path` 父目录） | 改成绝对路径或调整 yaml |
| 手柄按键无反应 | `USE_JOYSTICK: False` 没改 / 手柄没配对 | 改 yaml；机器人主电源重启重新连手柄 |

