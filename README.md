# Piper 机械臂启动及采集流程

## 一、安装项目

安装 lerobot 项目

```bash
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg=7.1.1 -c conda-forge

pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install torchcodec==0.2.1

git clone https://github.com/huggingface/lerobot.git
git checkout aec1b29
cd lerobot
pip install -e .
# 实际过程中如果报错rerun没有Scalar属性，则需rerun-sdk==0.22.1
```

安装本项目

```bash
cd ..
pip install -e .
```

安装机械臂所需的 can 口管理软件包

```bash
sudo apt update && sudo apt install can-utils ethtool
```

安装 piper 所需环境依赖以及 SDK

```bash
pip install python-can
pip install piper_sdk
```

安装 realsense SDK
```bash
pip install pyrealsense2
```

## 二、寻找并激活 piper 机械臂的 can 口

寻找设备连接的所有 can 模块

```bash
bash scripts/piper_can/find_all_can_port.sh
```

输出类似如下：\
Both ethtool and can-utils are installed. \
Interface can0 is connected to USB port 3-1.4:1.0 \
Interface can1 is connected to USB port 3-1.1:1.0

激活 can 设备。假设上面的 USB port 中 3-1.4:1.0 对应主臂，3-1.1:1.0 对应从臂，执行：

```bash
bash scripts/piper_can/can_activate.sh can_leader 1000000 "3-1.4:1.0"
bash scripts/piper_can/can_activate.sh can_follower 1000000 "3-1.1:1.0"
```

解释：接口名称需要重命名，以便与代码中进行对应；设定波特率为1000000(不可更改)
