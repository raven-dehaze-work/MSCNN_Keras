"""
main program for MSCNN-Dehazing
"""
import model

# 设置超参数
learning_rate = 1e-03
epochs = 20
batch_size = 32
# mode: 当前程序是train还是test。默认train
mode = 'train'      # or 'test'

