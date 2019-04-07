"""
main program for MSCNN-Dehazing
"""
from model import MSCNN
import numpy as np
import os
import utils
import matplotlib.pyplot as plt

# 设置超参数
learning_rate = 1e-04
epochs = 150
batch_size = 100

# 设置训练用的数据集图片大小
train_img_height = 230
train_img_width = 310

# mode: 当前程序是train还是test。默认train
mode = 'train'  # or 'test'

# 数据集合所占用比例
TRAIN_PERCENTAGE = 0.94
VAL_PERCENTAGE = 0.05

# 训练数据目录
train_haze_dir = './datasets/npy/hazy'
train_trans_dir = './datasets/npy/trans'


def load_data_generator(train_percentage, val_percentage):
    """
    根据各数据集合所占百分比来生成数据生成器。 分别有训练生成器，验证生成器，测试生成器（适合大数据量）
    :param train_percentage: 训练数据所占比例
    :param val_percentage: 验证数据所占比例
    :return: 训练数据字典，验证数据字典，测试数据字典的元祖 - (train_datas,val_datas,test_datas)
            每个字典有两个键: 1. 数据总量 nums
                            2. 该类数据生成器
    """
    if train_percentage + val_percentage >= 1:
        raise Exception()

    # 计算数据总量t
    trans_hazy_pair = utils.load_training_data_pair()
    haze_file_names = trans_hazy_pair['hazy']
    trans_file_names = trans_hazy_pair['trans']
    file_nums = len(haze_file_names)
    print("total file numbers is %d" % file_nums)

    # 分别计算训练数据，验证数据，测试数据需要的数量
    train_nums = int(file_nums * train_percentage)  # 会不会溢出？？，一般数据量还是没有这么大
    val_nums = int(file_nums * val_percentage)
    test_nums = file_nums - train_nums - val_nums
    print('the samples number of train. val and test are %d,%d,%d' % (train_nums, val_nums, test_nums))

    # 产生各类生成器
    train_generator = _get_generator(haze_file_names, trans_file_names, batch_size, 0, train_nums)
    val_generator = _get_generator(haze_file_names, trans_file_names, batch_size, train_nums, val_nums)
    test_generator = _get_generator(haze_file_names, trans_file_names, batch_size, train_nums + val_nums, test_nums)

    # 生成数据字典
    train_datas = {}
    train_datas['nums'] = train_nums
    train_datas['generator'] = train_generator

    val_datas = {}
    val_datas['nums'] = val_nums
    val_datas['generator'] = val_generator

    test_datas = {}
    test_datas['nums'] = test_nums
    test_datas['generator'] = test_generator

    return train_datas, val_datas, test_datas


def _get_generator(x_names, y_names, batch_size, start_pos, nums):
    """
    获取生成器
    :param x_names 数据的x数据文件名 -- haze file names
    :param y_names 数据的y数据文件名 -- trans file names
    :param batch_size 每次迭代生成多少个数据
    :param start_pos: 文件名中的偏移，起始位 即从file_names的start pos开始往后算train_nums个数据
    :param nums: 数据共有多少
    :return: 生成器
    """
    total_nums = len(x_names)
    if total_nums < start_pos + nums:
        raise Exception('start_pos + nums > total_nums, overflow the training data number')

    x_datas = np.zeros((batch_size, train_img_height, train_img_width, 3))
    y_datas = np.zeros((batch_size, train_img_height, train_img_width, 1))
    while True:
        for idx in range(nums):
            x_data = np.load(os.path.join(train_haze_dir, x_names[start_pos + idx])) / 255
            y_data = np.expand_dims(np.load(os.path.join(train_trans_dir, y_names[start_pos + idx])), 2) / 255

            x_datas[idx % batch_size] = x_data
            y_datas[idx % batch_size] = y_data
            if (idx + 1) % batch_size == 0:
                # 打印一下检验是否配对
                # print('\n',x_names[start_pos + idx],y_names[start_pos + idx])
                yield x_datas, y_datas
        if nums < batch_size:
            yield x_datas[0:nums], y_datas[0:nums]


def load_datas(train_percentage, val_percentage):
    """
    根据各数据集合所占百分比来加载数据集合（适合小数据量）
    :param train_percentage: 训练数据所占比例
    :param val_percentage: 验证数据所占比例
    :return: 训练数据，验证数据，测试数据所占比例的元祖 (train_datas, val_datas, test_datas)
            其中每个datas都是一个字典。键有 'haze' 和 'trans' 两个
    """
    if train_percentage + val_percentage >= 1:
        raise Exception("percentage of train and val overflow!!!")

    # 计算数据总量
    trans_hazy_pair = utils.load_training_data_pair()
    haze_file_names = trans_hazy_pair['hazy']
    trans_file_names = trans_hazy_pair['trans']

    file_nums = len(haze_file_names)
    print("total file numbers is %d" % file_nums)

    # 分别计算训练数据，验证数据，测试数据需要的数量
    train_nums = int(file_nums * train_percentage)  # 会不会溢出？？，一般数据量还是没有这么大
    val_nums = int(file_nums * val_percentage)
    test_nums = file_nums - train_nums - val_nums
    print('the samples number of train. val and test are %d,%d,%d' % (train_nums, val_nums, test_nums))

    # 开始正式加载
    train_datas = {}
    train_datas['haze'] = np.zeros((train_nums, train_img_height, train_img_width, 3))
    train_datas['trans'] = np.zeros((train_nums, train_img_height, train_img_width, 1))

    val_datas = {}
    val_datas['haze'] = np.zeros((val_nums, train_img_height, train_img_width, 3))
    val_datas['trans'] = np.zeros((val_nums, train_img_height, train_img_width, 1))

    test_datas = {}
    test_datas['haze'] = np.zeros((test_nums, train_img_height, train_img_width, 3))
    test_datas['trans'] = np.zeros((test_nums, train_img_height, train_img_width, 1))

    for idx, file_name in enumerate(haze_file_names):
        if idx < train_nums:
            train_datas['haze'][idx] = np.load(os.path.join(train_haze_dir, file_name))
            # 由于透射图是在保存时是2维，但是在训练时需要3维数据，所以需要补充一维
            train_datas['trans'][idx] = np.expand_dims(np.load(
                os.path.join(train_trans_dir, trans_file_names[idx])), 2)
        elif idx < train_nums + val_nums:
            val_datas['haze'][idx - train_nums] = np.load(os.path.join(train_haze_dir, file_name))
            val_datas['trans'][idx - train_nums] = np.expand_dims(np.load(
                os.path.join(train_trans_dir, trans_file_names[idx])), 2)
        else:
            test_datas['haze'][idx - train_nums - val_nums] = np.load(os.path.join(train_haze_dir, file_name))
            test_datas['trans'][idx - train_nums - val_nums] = np.expand_dims(np.load(
                os.path.join(train_trans_dir, trans_file_names[idx])), 2)

    # 打印测试是否转化成功
    print('train_datas shape', train_datas['haze'].shape)
    print('val_datas shape', val_datas['haze'].shape)
    print('test_datas shape', test_datas['haze'].shape)

    print('train_datas shape', train_datas['trans'].shape)
    print('val_datas shape', val_datas['trans'].shape)
    print('test_datas shape', test_datas['trans'].shape)
    return (train_datas, val_datas, test_datas)


def get_airlight(trans_map, J):
    """
    根据透射图和雾图，得到大气光
    :param trans_map:
    :return:
    """
    pixel_num = trans_map.shape[0] * trans_map.shape[1]
    # 取得前1%的最小像素点
    fractor = 0.01
    # 待选数
    selected_num = int(pixel_num * fractor)

    # 先给透射图排个序,方便统计
    trans_map_sorted = np.sort(trans_map)

    # 获得索引
    (row_idx, col_idx) = np.where(trans_map_sorted[0:selected_num] == trans_map)

    A = np.max(J[row_idx, col_idx])
    return A


if __name__ == '__main__':
    # 建立模型
    mscnn = MSCNN(batch_size, epochs, learning_rate)
    # 载入训练数据
    (train_datas, val_datas, test_datas) = load_data_generator(TRAIN_PERCENTAGE, VAL_PERCENTAGE)

    if mode == 'train':
        # train
        mscnn.train_on_generator(train_datas, val_datas)
    elif mode == 'test':
        # test
        mscnn.test_on_generator(test_datas)
