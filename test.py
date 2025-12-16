import os
import cv2
import math
import numpy as np
import keras
from keras import backend as K
from mymodule import deeplearning_IO as dio
from mymodule import cnn_models as cm
import matplotlib

matplotlib.use('Agg')

# === 全局配置 ===
W = 32
K_ROT = 10
SCALE = 1


def load_model_once(weights_path='weight_and_model_user.hdf5'):
    print(f"🏗️  Loading model from {weights_path}...")
    if K.image_data_format() == 'channels_first':
        model = cm.get_densenet_2d_channel_first_2dense(W, W)
    else:
        model = cm.get_densenet_2d_channel_last_2dense(W, W)
    model.load_weights(weights_path)
    return model


def test_one_object(model, obj_full_path, output_dir=None):
    """
    测试单个物体：预测、融合、算误差、保存图片
    """
    # 获取物体名字 (用于文件名)
    obj_name = os.path.basename(obj_full_path.rstrip('/'))

    # 1. 准备数据 (调用 dio)
    try:
        # dio 会自动处理 bearPNG (只要不传 index)
        [Sv, Nv, Rv, IDv, Szv] = dio.prep_data_2d_from_images_test([obj_full_path], SCALE, W, K_ROT)
    except Exception as e:
        print(f"❌ Data load failed for {obj_name}: {e}")
        return None

    # 提取单张数据
    S, N_gt_flat, R, ID = Sv[0], Nv[0], Rv[0], IDv[0]
    height, width = Szv[0, 0], Szv[0, 1]
    rotdiv = S.shape[1]

    # 2. 预测循环 (Test Time Augmentation)
    NestList = []

    # print(f"   Running {rotdiv} rotations...")
    for r in range(rotdiv):
        # 取出第 r 个旋转角度的数据
        embed_div = S[:, r, :, :]

        # 适配维度
        if K.image_data_format() == 'channels_last':
            embed_div = np.reshape(embed_div, (-1, W, W, 1))
        else:
            embed_div = np.reshape(embed_div, (-1, 1, W, W))

        # --- 核心预测 ---
        outputs = model.predict(embed_div)

        # --- 反向旋转 (Rotate Back) ---
        # 构造反旋转矩阵
        rot_mat = R[r, :, :]  # (2, 2)
        inv_rot = np.linalg.inv(rot_mat)

        # 此时 outputs 是 (Batch, 3) -> (nx, ny, nz)
        # 我们只旋转 xy，z 不变
        n_xy = outputs[:, 0:2].T  # (2, Batch)
        n_xy_back = np.dot(inv_rot, n_xy)  # (2, Batch)

        # 拼回 (nx, ny, nz)
        n_rotated = np.zeros_like(outputs)
        n_rotated[:, 0] = n_xy_back[0, :]
        n_rotated[:, 1] = n_xy_back[1, :]
        n_rotated[:, 2] = outputs[:, 2]

        # 存入列表等待平均
        NestList.append(n_rotated)

    # 3. 融合 (Vector Average)
    # 形状: (10, Batch, 3) -> Mean -> (Batch, 3)
    NestMean = np.mean(np.array(NestList), axis=0)

    # 归一化
    norm = np.sqrt(np.sum(NestMean ** 2, axis=1, keepdims=True))
    NestMean = NestMean / (norm + 1e-6)

    # 4. 计算误差 (MAE)
    # 取出真值 (N_gt_flat 已经包含了旋转维度 [Batch, 10, 3]，我们取第0个角度作为 GT)
    # 注意：dio 返回的 Nv 形状通常是 (Batch, RotDiv, 3) 或者 (Batch, 3)
    # 根据之前的报错经验，这里直接取数据即可，不需要 ID 索引，因为 Nv 已经是 Valid Set
    if N_gt_flat.ndim == 3:
        gt_valid = N_gt_flat[:, 0, :]  # 取第0个旋转角度
    else:
        gt_valid = N_gt_flat  # 如果只有2维，直接用

    # 计算点积 -> 角度
    dot = np.sum(NestMean * gt_valid, axis=1)
    dot = np.clip(dot, -1.0, 1.0)
    err_deg = np.arccos(np.abs(dot)) * 180.0 / np.pi
    mae = np.mean(err_deg)

    # 5. 生成并保存图片
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 还原全尺寸法线图
        full_pred = np.zeros((height * width, 3), dtype=np.float32)
        full_pred[ID] = NestMean
        img_pred = full_pred.reshape((height, width, 3))

        # 还原全尺寸误差图
        full_err = np.zeros((height * width), dtype=np.float32)
        full_err[ID] = err_deg
        img_err = full_err.reshape((height, width))

        # 可视化处理
        # 法线: [-1,1] -> [0,255]
        vis_n = np.clip(127 * (img_pred + 1), 0, 255).astype(np.uint8)
        # 误差: 放大5倍显示 (0~50度映射为0~255)
        vis_e = np.clip(5 * np.dstack([img_err] * 3), 0, 255).astype(np.uint8)

        # 拼接: 左法线，右误差
        concat = np.concatenate((vis_n, vis_e), axis=1)

        # 保存
        save_name = f"{obj_name}_MAE_{mae:.2f}.png"
        save_path = os.path.join(output_dir, save_name)
        cv2.imwrite(save_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
        # print(f"   💾 Saved: {save_name}")

    return mae


if __name__ == '__main__':
    # 单独运行时的逻辑
    model = load_model_once()
    test_one_object(model, '/data/qsy/Datasets/DiLiGenT/pmsData/ballPNG', './test_results_single')