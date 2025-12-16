#!/bin/bash

# 1. 强制加载 NVIDIA 库 (解决 undefined symbol 报错)
export ENV_LIB_PATH=/home/ubuntu/anaconda3/envs/sps_CNN-PS/lib/python3.8/site-packages/nvidia
export LD_LIBRARY_PATH=$ENV_LIB_PATH/cublas/lib:$ENV_LIB_PATH/cudnn/lib:$ENV_LIB_PATH/cusolver/lib:$ENV_LIB_PATH/cusparse/lib:$ENV_LIB_PATH/nvjitlink/lib:$LD_LIBRARY_PATH

# 2. 打印提示
echo "开始在 RTX 3090 上训练 CNN-PS..."

# 3. 运行训练脚本
python train.py