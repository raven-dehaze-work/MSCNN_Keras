"""
测试文件--用于开发过程中的各项功能测试
1. 所有npy 文件均为255 文件
2. hazy文件的末尾数字为大气光
3. keras的模型save和load是有效的
4. reside ITS数据库 trans和hazy是相对应的
"""

import utils
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.models import Sequential,Model
from keras.layers import Dense,Input

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
    # 随机生成x和y
    x = np.linspace(-1,1,100*1000).reshape((100*1000,-1))
    y = 0.5 * x + 0.1 + np.random.normal(0,0.01,x.shape)
    x_test = [0.4]

    # 构建MLP网络
    z = Input((1,))
    layer1 = Dense(500)(z)
    out = Dense(1)(layer1)
    model = Model(inputs=z,outputs=out)

    # 编译网络
    model.summary()
    model.compile(optimizer='sgd', loss='mse')

    # 训练
    # model.fit(x,y,256,epochs=10)

    # 保存
    # model.save('mlp_weight.h5')
    model.load_weights('mlp_weight.h5')
    # 预测
    print(model.predict(x_test))