"""
some utils function to help the process of training
"""
import os
import numpy as np
def load_training_data_pair():
    # testing dir where locate the training datas
    read_dir1 = './datasets/npy/trans'
    read_dir2 = './datasets/npy/hazy'

    # load file names
    trans_names = [file for root, dirs, file in os.walk(read_dir1)][0]
    hazy_names = [file for root,dirs,file in os.walk(read_dir2)][0]
    # get the number of total files
    file_nums = len(trans_names)

    # remove suffix
    trans_without_suffix = list(map(
        lambda x: x.split('.')[0]+'_',      # add '_' char to match hazy_name
        trans_names
    ))

    # declare pairs var to save trans and hazy which are matched
    pairs = {}
    pairs['trans'] = ['' for i in range(file_nums)]
    pairs['hazy'] = ['' for i in range(file_nums)]

    # design a function to match trans_names and hazy_names to form a pair so that can be used to train the network
    count = 0
    for idx,trans_name in enumerate(trans_without_suffix):
        for hazy_name in hazy_names:
            if trans_name in hazy_name:
                pairs['trans'][idx] = trans_names[idx]
                pairs['hazy'][idx] = hazy_names[idx]

                count += 1
                if count % 1000 == 0:
                    print('trans %s hazy %s' % (trans_names[idx], hazy_names[idx]))
                break

    # print match pair info
    print('the number of matched paired is %d, and total file number is %d' % (count,file_nums))
    return pairs

if __name__ == '__main__':
    # 所有npy文件全是255图片
    print(np.load('./datasets/npy/trans/1_1.npy'))
    print(np.load('./datasets/npy/hazy/790_9_0.75398.npy'))
    print(np.load('./datasets/npy/clear/1.npy'))
