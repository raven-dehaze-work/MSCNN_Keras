"""
测试文件--用于开发过程中的各项功能测试
"""

import utils
import matplotlib.pyplot as plt
import os
import numpy as np


def _get_airlight(file_name):
    """
    从数据文件名中获取airlight的值。（reside数据库中的文件名给出了airlight的值）
    :param file_name: 雾图文件名
    :return:
    """
    s_idx = file_name.find('.')
    e_idx = file_name[s_idx+1:].find('.')
    return float(file_name[s_idx-1:e_idx+s_idx+1])

if __name__ == '__main__':
    """
    验证数据集的trans和hazy还原效果
    """
    trans_dir = './datasets/npy/trans'
    hazy_dir = './datasets/npy/hazy'

    trans_hazy_pairs = utils.load_training_data_pair()
    trans = trans_hazy_pairs['trans']
    hazys = trans_hazy_pairs['hazy']

    # 删除 pairs 释放内存
    del trans_hazy_pairs

    # 要测试第几张图片
    num_idx = 13000

    # 测试一张图片
    test_trans_file_path = os.path.join(trans_dir,trans[num_idx])
    test_hazy_file_path = os.path.join(hazy_dir,hazys[num_idx])

    print('test trans file name %s' % test_trans_file_path)
    print('test hazy file name %s' % test_hazy_file_path)

    tran_img = np.load(test_trans_file_path)/255
    hazy_img = np.load(test_hazy_file_path)/255

    # 根据还原公式进行还原并显示三张图
    clear_img = np.zeros(hazy_img.shape)

    A = _get_airlight(hazys[num_idx])
    print('airlight value %f' % A)

    # 还原
    clear_img[:,:,0] = (hazy_img[:,:,0] - A)/tran_img + A
    clear_img[:,:,1] = (hazy_img[:,:,1] - A)/tran_img + A
    clear_img[:,:,2] = (hazy_img[:,:,2] - A)/tran_img + A

    # show img
    plt.subplot(1,2,1)
    plt.imshow(hazy_img)
    plt.subplot(1,2,2)
    plt.imshow(clear_img)
    plt.show()

