# Copyright 2018, Satoshi Ikehata, National Institute of Informatics (sikehata@nii.ac.jp)
w = 32 # size of observation map

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
from mymodule import cnn_models as cm
import gc  # <--- 添加这一行

isParallel = False  # 改为单卡模式 # one if multi gpu is available

if K.image_data_format() == 'channels_first': # theano
    model = cm.get_densenet_2d_channel_first_2dense(w,w)
else: # tensorflow or cntk
    model = cm.get_densenet_2d_channel_last_2dense(w,w)
if isParallel:
    parallel_model = multi_gpu_model(model, gpus=3)
    parallel_model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')
else:
    model.compile(optimizer=keras.optimizers.Adam(), loss='mean_squared_error')

datagenerated = '/data/sps/datasets/CyclesPSDataset/CyclesSpecularMetallic' # path to the training dataset
objlist = sorted(os.listdir(datagenerated + '/PRPS'))
epochs = 1
min_err = 1000
rotdivin = 10
rotdivon = 10
# --- 核心修改区 ---
loopnum = 10   # 显式定义总轮数 (原代码里是硬写的 range(10))
divnum = 15    # 将原来的 datasplit 改名为 divnum，并设置为 15 (对应15个物体)

# 计算每个分块的大小 (如果是 15/15，结果就是 1)
subsetsize = np.int32(len(objlist) / divnum)

# 外层循环：k 代表当前的轮数 (对应我之前说的 loop)
for k in range(loopnum):
    datalist = []
    # 内层循环：p 代表当前的分块 (对应我之前说的 k)
    for p in range(divnum):
        # ================= 修正后的打印代码 =================
        # 变量映射关系：
        # loop (当前轮) -> k
        # loopnum (总轮数) -> loopnum
        # divnum (总块数) -> divnum
        # k (当前块) -> p  <-- 注意这里！内层循环变量其实是 p

        print('\n' + '=' * 60)
        # 这里的 k+1 是当前第几轮，p+1 是当前第几块
        print(f'🚀 进度指示: 正在运行第 {k + 1} 轮 (共 {loopnum} 轮) | 第 {p + 1} 个分块 (共 {divnum} 块)')
        print('=' * 60 + '\n')
        # ==================================================
        SList = []
        NList = []
        for q in range(subsetsize):

            objroot = datagenerated + '/PRPS_Diffuse'
            dirname = 'images_diffuse'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print(datapath)
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 1, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

            objroot = datagenerated + '/PRPS'
            dirname = 'images_specular'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print("images_specular=" + " " + datapath)
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 0.5, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

            objroot = datagenerated + '/PRPS'
            datapath = [objroot + '/' + '%s' % objlist[subsetsize*p+q]]
            print("images_metallic=" + " " + datapath)
            dirname = 'images_metallic'
            S,M,N = dio.prep_data_2d_from_images_cycles(datapath, dirname, 0.5, w, rotdivin, rotdivon)
            SList.append(S.copy())
            NList.append(N.copy())
            del S, M, N

        SList = np.float32(np.concatenate(SList, axis=0))
        NList = np.float32(np.concatenate(NList, axis=0))

        if  isParallel:
            hist = parallel_model.fit(SList, NList, batch_size= 768, epochs= epochs, verbose=1, shuffle=True, validation_split=0.1)
        else:
            hist = model.fit(SList, NList, batch_size= 1024, epochs= epochs, verbose=1, shuffle=True, validation_split=0.1)

        model.save('weight_and_model_user.hdf5')
        print('Model Updated!!')

        # --- 修改后的代码 (带显式验证) ---
        del SList, NList

        # 让 gc 跑，并用 n 记录它回收了多少垃圾
        n = gc.collect()

        print(f'🧹 [内存大扫除] 本轮清理了 {n} 个废弃对象，内存已释放！')
        print('-' * 60)