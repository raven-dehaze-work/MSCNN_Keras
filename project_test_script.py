"""
辅助脚本，用于测试一些编程中的功能，不重要
"""

from random import choice
import glob
import os
import numpy as np


img_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_img'
npy_dir = r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy'

# 读取clear目录下的所有文件名
clear_file_names = glob.glob(os.path.join(img_dir,'clear/*'))
# 去除所有后缀
clear_file_names = list(map(lambda x:os.path.splitext(
    os.path.basename(x)
)[0],clear_file_names))

# 循环去除多余文件
for file_name in clear_file_names:
    # 寻找trans匹配对
    trans_file_names = glob.glob(os.path.join(img_dir,'trans',file_name+"_*"))
    # 随机选择一个要剩下的
    trans_remained_file = choice(trans_file_names)

    # 删除其余所有不需要的
    for file in trans_file_names:
        if trans_remained_file != file:
            print(file)
            os.remove(file)

    # 删除haze file
    remained_prefix = os.path.splitext(os.path.basename(trans_remained_file))[0]
    # 寻找haze匹配对
    haze_file_names = glob.glob(os.path.join(img_dir,'haze',file_name+"_*"))
    # 用remained_prefix 去匹配 寻找到的haze_file匹配对，所有不匹配的全部删除
    for file in haze_file_names:
        if not remained_prefix + "_" in file:
            print(file)
            os.remove(file)

    # 删除npy对应的文件
    # 删除haze
    npy_haze_file_names = glob.glob(os.path.join(npy_dir,'haze',file_name+"_*"))
    # 删除所有不需要的
    for file in npy_haze_file_names:
        if not remained_prefix+"_" in file:
            print(file)
            os.remove(file)

    # 删除trans
    npy_trans_file_names = glob.glob(os.path.join(npy_dir, 'trans', file_name + "_*"))
    # 删除所有不需要的
    for file in npy_trans_file_names:
        if not remained_prefix+"." in file:
            print(file)
            os.remove(file)