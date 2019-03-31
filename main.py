"""
main program for MSCNN-Dehazing
"""
import model
import numpy as np

# 设置超参数
learning_rate = 1e-03
epochs = 20
batch_size = 32
# mode: 当前程序是train还是test。默认train
mode = 'train'      # or 'test'


def get_airlight(trans_map,J):
    """
    根据透射图和雾图，得到大气光
    :param trans_map:
    :return:
    """
    pixel_num = trans_map.shape[0]*trans_map.shape[1]
    # 取得前1%的最小像素点
    fractor = 0.01
    # 待选数
    selected_num = int(pixel_num*fractor)

    # 先给透射图排个序,方便统计
    trans_map_sorted = np.sort(trans_map)

    # 获得索引
    (row_idx,col_idx) = np.where(trans_map_sorted[0:selected_num] == trans_map)

    A = np.max(J[row_idx,col_idx])
    return A
