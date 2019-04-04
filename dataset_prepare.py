# -*- coding: utf-8 -*-
"""
准备数据集合脚本
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import h5py
import os
from PIL import Image
import matplotlib.pyplot as plt
from main import *


def extract_src_imgs():
    """
    提取原图
    :return:
    """
    f = h5py.File("nyu_depth_v2_labeled.mat")
    images = f["images"]
    images = np.array(images)

    path_converted = './datasets/image/clear'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    images_number = []
    for i in range(len(images)):
        print(str(i) + '.png')
        images_number.append(images[i])
        a = np.array(images_number[i])

        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        img = img.resize((train_img_width,train_img_height))


        iconpath = os.path.join(path_converted, str(i) + '.png')
        img.save(iconpath, optimize=True)
    f.close()

def extract_depth_imgs():
    """
    提取深度图
    :return:
    """
    f = h5py.File("nyu_depth_v2_labeled.mat")
    depths = f["depths"]
    depths = np.array(depths)

    path_converted = './datasets/image/depth'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    max = depths.max()
    depths = depths / max * 255
    depths = depths.transpose((0, 2, 1))

    for i in range(len(depths)):
        print(str(i) + '.png')
        depths_img = Image.fromarray(np.uint8(depths[i]))
        depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)
        depths_img = depths_img.resize((train_img_width,train_img_height))

        iconpath = os.path.join(path_converted, str(i) + '.png')
        depths_img.save(iconpath, optimize=True)
    f.close()


def generate_trans():
    """
    生成透射图
    :return:
    """
    depth_dir = './datasets/image/depth'
    trans_dir = './datasets/image/trans'

    # 获取所有景深图形文件名
    file_names = [file for root, dirs, file in os.walk(depth_dir)][0]
    file_num = len(file_names)
    # 介质消光系数
    betas = np.random.rand(file_num) + 0.5

    for idx, file_name in enumerate(file_names):
        print(file_name)
        # 得到图像的灰阶图
        img = Image.open(os.path.join(depth_dir, file_name))
        img_arr = np.array(img) / 255

        # 生成透射图
        trans_img = np.exp(-betas[idx] * img_arr)

        # 保存
        trans_img = Image.fromarray(np.uint8(trans_img * 255))
        trans_img.save(os.path.join(trans_dir, file_name))


def generate_clear_npy():
    """
    生成清晰图像的npy文件，方便训练时直接提取
    :return:
    """
    clear_dir = './datasets/image/clear'
    npy_clear_dir = './datasets/npy/clear'

    file_names = [file for root, dirs, file in os.walk(clear_dir)][0]

    for idx, file_name in enumerate(file_names):
        print(file_name)
        img = Image.open(os.path.join(clear_dir, file_name))
        img = img.resize((train_img_width,train_img_height))
        img_arr = np.array(img)
        np.save(os.path.join(npy_clear_dir, file_name.replace('.png','.npy')), img_arr)


def generate_trans_npy():
    """
    生成透射图的npy文件，方便训练时直接提取
    :return:
    """
    trans_dir = './datasets/image/trans'
    npy_trans_dir = './datasets/npy/trans'

    file_names = [file for root, dirs, file in os.walk(trans_dir)][0]

    for idx, file_name in enumerate(file_names):
        print(file_name)
        img = Image.open(os.path.join(trans_dir, file_name))
        img = img.resize((train_img_width, train_img_height))
        img_arr = np.array(img)
        np.save(os.path.join(npy_trans_dir, file_name.replace('.png', '.npy')), img_arr)


def generate_haze_img_npy():
    """
    生成带雾图的png图片和npy文件 -- 适用于有清晰图和透射图的时候进行合成
    :return:
    """
    npy_clear_dir = './datasets/npy/clear'
    npy_trans_dir = './datasets/npy/trans'
    npy_haze_dir = './datasets/npy/hazy'
    npy_haze_image_dir = './datasets/image/hazy'

    clear_file_names = [file for root, dirs, file in os.walk(npy_clear_dir)][0]
    trans_file_name = [file for root, dirs, file in os.walk(npy_trans_dir)][0]

    for idx, file_name in enumerate(clear_file_names):
        print(file_name)
        clear_arr = np.load(os.path.join(npy_clear_dir, clear_file_names[idx])) / 255
        trans_arr = np.load(os.path.join(npy_trans_dir, trans_file_name[idx])) / 255

        A = np.random.rand() * 0.3 + 0.7

        I = np.zeros((clear_arr.shape))
        I[:, :, 0] = clear_arr[:, :, 0] * trans_arr + A * (1 - trans_arr)
        I[:, :, 1] = clear_arr[:, :, 1] * trans_arr + A * (1 - trans_arr)
        I[:, :, 2] = clear_arr[:, :, 2] * trans_arr + A * (1 - trans_arr)

        # 保存为npy
        np.save(os.path.join(npy_haze_dir, file_name), np.uint8(I * 255))
        haze_img = Image.fromarray(np.uint8(I * 255))
        # 保存图片
        haze_img.save(os.path.join(npy_haze_image_dir, file_name.replace('.npy','.png')))

def genernate_hazy_npy():
    """
    生成雾图的npy文件，方便后续训练时直接读取
    :return:
    """
    hazy_img_file_dir ='./datasets/image/hazy'
    hazy_npy_file_dir = './datasets/npy/hazy'

    hazy_img_file_names = [file_name for root,dir,file_name in os.walk(hazy_img_file_dir)][0]

    for idx, file_name in enumerate(hazy_img_file_names):
        print(file_name)
        img = Image.open(os.path.join(hazy_img_file_dir, file_name))
        img = img.resize((train_img_width, train_img_height))
        img_arr = np.array(img)
        np.save(os.path.join(hazy_npy_file_dir, file_name.replace('.png', '.npy')), img_arr)

def generate_imgs():
    """
    生成 清晰图，景深图，透射图
    :return:
    """
    extract_src_imgs()
    extract_depth_imgs()
    generate_trans()

def generate_npys():
    """
    将 清晰图，景深图，透射图的图片文件转为npy文件，方便直接读取
    :return:
    """
    generate_clear_npy()
    generate_trans_npy()
    genernate_hazy_npy()

if __name__ == '__main__':
    # generate_imgs()
    generate_npys()

    # 单独测试
    # generate_clear_npy()
    # generate_trans_npy()
    # generate_haze_img_npy()