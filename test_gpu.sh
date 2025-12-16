#!/bin/bash

# 1. 设置专属的 GPU 库路径 (只在运行这个脚本时生效)
export ENV_LIB_PATH=/home/ubuntu/anaconda3/envs/sps_CNN-PS/lib/python3.8/site-packages/nvidia
export LD_LIBRARY_PATH=$ENV_LIB_PATH/cublas/lib:$ENV_LIB_PATH/cudnn/lib:$ENV_LIB_PATH/cusolver/lib:$ENV_LIB_PATH/cusparse/lib:$ENV_LIB_PATH/nvjitlink/lib:$LD_LIBRARY_PATH

# 2. 打印提示
echo "GPU 环境配置完成，正在使用 RTX 3090 运行..."

# 3. 运行 Python 代码
python test_all.py