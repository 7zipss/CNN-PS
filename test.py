# Copyright 2018, Satoshi Ikehata, National Institute of Informatics (sikehata@nii.ac.jp)

import importlib
import numpy as np
import pydot
import os
import keras
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from keras.utils import multi_gpu_model
from mymodule import deeplearning_IO as dio
from mymodule import cnn_models as cm  # <---【新增这一行】
from PIL import Image
import matplotlib
matplotlib.use('Agg') # 强制不显示窗口，只保存图片【新增】防止在服务器上报错
import matplotlib.pyplot as plt

# NOTICE!!
# bearPNG has problem in first 20 images, so please discard first 20 information from Sv Nv Rv IDv Szv if you want the result from the paper
# harvestPNG has problem in the order of values in normal.txt, so you need flip the surface normal map upside down

def main():
    isParallel = False # one if multi gpu is available
    dirlist = []
    # 测试数据集
    diligent = '/data/qsy/Datasets/DiLiGenT/pmsData'

    # Set the directory list
    # objlist = ('ballPNG','bearPNG','buddhaPNG','catPNG','cowPNG','gobletPNG','harvestPNG','pot1PNG','pot2PNG','readingPNG')
    objlist = (['bearPNG'])



    for f in objlist:
        dirlist.append(diligent + '/' + f)
    print(dirlist)

    # Prepare images
    scale = 1 # downsize the image by this scale
    w = 32 # size of observation map
    # 【修改点 1】将 K 改名为 K_rot，避免覆盖 keras.backend 的 K
    K_rot = 10 # the number of different rotations for the rotational pseudo-invariance

    # 【修改点 2】在调用函数时，使用新的变量名 K_rot
    # [Sv, Nv, Rv, IDv, Szv] = dio.prep_data_2d_from_images_test(dirlist, scale, w, K_rot) # Comment this line when runnning test on bearPNG
    [Sv, Nv, Rv, IDv, Szv] = dio.prep_data_2d_from_images_test(dirlist, scale, w, K_rot, index = range(20, 96)) # Uncomment this if you want to use the subset of images (in this case, 20-th to 96-th images are input)
    # Load pretrained model
    #model = load_model('weight_and_model.hdf5') 注释并替换为下面修改内容
    # ------------------ 修改开始 ------------------
    # 由于 Python 版本不同 (3.5 vs 3.8)，直接 load_model 会报 bad marshal data
    # 解决方案：手动构建模型结构，然后只加载权重

    # 根据 train.py 的逻辑，w=32
    w = 32

    # 判断数据格式 (TensorFlow 默认通常是 channels_last)
    if K.image_data_format() == 'channels_first':
        print("Building model (channels_first)...")
        model = cm.get_densenet_2d_channel_first_2dense(w, w)
    else:
        print("Building model (channels_last)...")
        model = cm.get_densenet_2d_channel_last_2dense(w, w)

    # 加载权重
    print("Loading weights...")
    model.load_weights('weight_and_model_user.hdf5')
    # ------------------ 修改结束 ------------------

    # Test network
    dio.TestNetwork(model,Sv,Nv,Rv,IDv,Szv,showFig=0,isTensorFlow=1) # showFig=0：关闭实时显示

if __name__ == '__main__':
    main()
