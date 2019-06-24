"""
数据加载脚本，用于辅助训练
"""
import os
import glob
import numpy as np


class DataLoader:
    """
    数据加载类
    """

    def __init__(self, batch_size=100) -> None:
        super().__init__()
        # 加载各数据生成器

        self.train_generator = self._build_data_generator(batch_size)
        self.train_size = len(
            glob.glob(os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\haze', '*')))
        self.test_size = len(
            glob.glob(os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\test_datasets_npy\haze', '*')))

    def _build_data_generator(self, batch_size):
        # 加载所有文件名
        haze_file_names = glob.glob(
            os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\haze', '*'))
        trans_file_names = glob.glob(
            os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\train_datasets_npy\trans', '*'))

        # 因为读入的haze 和 trans 文件目前还不是匹配的pair，所以做一下匹配处理
        pairs_num = len(haze_file_names)
        pairs = {'haze': ['' for i in range(pairs_num)],
                 'trans': ['' for i in range(pairs_num)]}

        # 匹配
        for idx, file_name in enumerate(trans_file_names):
            pairs['trans'][idx] = file_name
            prefix = os.path.splitext(
                os.path.basename(file_name)
            )[0]
            for haze_file_name in haze_file_names:
                if prefix + "_" in haze_file_name:
                    pairs['haze'][idx] = haze_file_name
                    haze_file_names.remove(haze_file_name)
                    break

        x_datas = np.zeros((batch_size, 240, 320, 3))
        y_datas = np.zeros((batch_size, 240, 320, 1))
        pairs['haze'] = np.array(pairs['haze'])
        pairs['trans'] = np.array(pairs['trans'])
        # 产生生成器
        while True:
            # 随机扰乱
            permutated_indexes = np.random.permutation(pairs_num)
            if (pairs_num < batch_size):
                # 所有数据量不足以提供一个batch_size
                x_paths = pairs['haze'][permutated_indexes]
                y_paths = pairs['trans'][permutated_indexes]
                for idx in range(pairs_num):
                    x_datas[idx] = np.load(x_paths[idx])
                    y_datas[idx] = np.load(y_paths[idx])
                yield x_datas, y_datas
            else:
                # 数量能够提供
                for index in range(pairs_num // batch_size):
                    batch_indexes = permutated_indexes[index * batch_size:(index + 1) * batch_size]

                    # batch_pairs仅仅是些文件路径
                    x_paths = pairs['haze'][batch_indexes]
                    y_paths = pairs['trans'][batch_indexes]

                    # 现在要把这些文件路径对应的npy文件读取成npy arr
                    for idx in range(batch_size):
                        x_datas[idx] = np.load(x_paths[idx])
                        y_datas[idx] = np.load(y_paths[idx]).reshape((240,320,1))

                    yield x_datas, y_datas

    def load_test_datasets(self, size):
        """
        加载测试数据集合。
        :param size 加载多少张测试图片，测试图片包括 haze图和trans图（后者用于对比）
        :return: haze,trans 元祖
        """
        choose_idx = np.random.randint(0, self.test_size, size)
        trans_paths = glob.glob(os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\test_datasets_npy\trans', '*'))
        haze_paths = glob.glob(os.path.join(r'D:\Projects\PythonProjects\MSCNN_Keras\test_datasets_npy\haze', '*'))
        trans = np.zeros((size, 240, 320, 1))
        haze = np.zeros((size, 240, 320, 3))

        for idx in range(size):
            trans[idx] = np.load(trans_paths[choose_idx[idx]]).reshape((240,320,1))
            haze[idx] = np.load(haze_paths[choose_idx[idx]])
        return haze, trans
if __name__ == '__main__':
    DataLoader()