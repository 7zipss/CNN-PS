import os
import csv
import numpy as np
import test  # 导入上面的 test.py

# === 配置 ===
DILIGENT_ROOT = '/data/qsy/Datasets/DiLiGenT/pmsData'
RESULT_DIR = './test_results_all'  # 结果保存总目录
WEIGHTS_PATH = 'weight_and_model_user.hdf5'

OBJ_LIST = ['ballPNG', 'bearPNG', 'buddhaPNG', 'catPNG', 'cowPNG',
            'gobletPNG', 'harvestPNG', 'pot1PNG', 'pot2PNG', 'readingPNG']


def main():
    # 1. 加载模型
    model = test.load_model_once(WEIGHTS_PATH)

    # 2. 准备结果目录
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    summary = []
    print(f"\n{'Object':<15} | {'MAE':<10}")
    print("-" * 30)

    # 3. 循环测试
    for obj_name in OBJ_LIST:
        full_path = os.path.join(DILIGENT_ROOT, obj_name)

        # 为每个物体创建一个子文件夹 (可选，如果想都堆在一起，就传 RESULT_DIR)
        # 这里演示：都存在 test_results_final/ 下，文件名带前缀

        # === 调用 test.py ===
        mae = test.test_one_object(model, full_path, output_dir=RESULT_DIR)

        if mae is not None:
            print(f"{obj_name:<15} | {mae:.4f}")
            summary.append({'Object': obj_name, 'MAE': mae})
        else:
            print(f"{obj_name:<15} | Failed")

    # 4. 统计与保存 CSV
    if summary:
        avg_mae = np.mean([x['MAE'] for x in summary])
        print("-" * 30)
        print(f"{'AVERAGE':<15} | {avg_mae:.4f}")

        csv_path = os.path.join(RESULT_DIR, 'summary_mae.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Object', 'MAE'])
            writer.writeheader()
            writer.writerows(summary)
            writer.writerow({'Object': 'AVERAGE', 'MAE': avg_mae})
        print(f"\n📄 Summary saved: {csv_path}")


if __name__ == '__main__':
    main()