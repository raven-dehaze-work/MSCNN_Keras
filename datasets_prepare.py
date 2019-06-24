"""
数据集准备脚本，实现功能
1. 从ITS文件夹中读取相应数据集
2. 将所有图片转为320*240的大小并存为npy文件
"""
import glob
import numpy as np
import os
from PIL import Image


# 定义目录参数
# 原始数据集合
clear_img_dir = r'D:\Projects\Dehaze\数据集合\ITS\clear'
hazy_img_dir = r'D:\Projects\Dehaze\数据集合\ITS\hazy'
trans_img_dir = r'D:\Projects\Dehaze\数据集合\ITS\trans'

# 将原始图片放缩后存放的图片目录
clear_scaled_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_img\clear'
hazy_scaled_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_img\haze'
trans_scaled_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_img\trans'

# 将原始图片放缩后的npy文件存放目录
clear_npy_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\clear'
hazy_npy_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\haze'
trans_npy_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\trans'

# 定义缩放后的图片大小
scaled_img_width = 320
scaled_img_height = 240


def convert_img2npy(img_dir, npy_dir,save_scaled=False,scaled_dir=None):
    """
    将图片缩放并转换为npy文件，方便后续训练使用 仅转换.png后缀
    :param img_dir: 图片目录
    :param npy_dir: 保存的npy文件目录
    :param save_scaled: 是否需要保存缩放后的图片
    :param scaled_dir: 缩放后保存的目录
    :return:
    """
    imgs_path = glob.glob(os.path.join(img_dir, "*.png"))

    for filepath in imgs_path:
        file_name = os.path.basename(filepath)
        print('convert %s' % file_name)

        # 打开图片
        img = Image.open(filepath)
        img = img.resize((scaled_img_width, scaled_img_height))

        # 是否保存sacled的图片
        if save_scaled and scaled_dir:
            img.save(os.path.join(scaled_dir, file_name))

        # 转为npy
        img_np = np.array(img)
        # 保存
        np.save(os.path.join(npy_dir, file_name.replace("png", "npy")), img_np)


if __name__ == '__main__':
    convert_img2npy(clear_img_dir,clear_npy_dir,True,clear_scaled_dir)
    convert_img2npy(hazy_img_dir,hazy_npy_dir,True,hazy_scaled_dir)
    convert_img2npy(trans_img_dir,trans_npy_dir,True,trans_scaled_dir)
