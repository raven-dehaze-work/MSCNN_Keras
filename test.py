"""
test script to test some functinos
"""

import utils
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == '__main__':
    trans_root_dir = './datasets/npy/trans'
    hazy_root_dir = './datasets/npy/hazy'
    trans_hazy_pairs = utils.load_training_data_pair()

    trans_file_names = trans_hazy_pairs['trans']
    hazy_file_names = trans_hazy_pairs['hazy']

    # plot a pair of trans and hazy image
    r_idx = np.random.random_integers(0,len(trans_file_names)-100,(1,))[0]
    trans = np.load(os.path.join(trans_root_dir,trans_file_names[r_idx]))
    hazy = np.load(os.path.join(hazy_root_dir,hazy_file_names[r_idx]))

    # show
    plt.subplot(1, 2, 1)
    plt.imshow(trans,cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(hazy)
    plt.show()
