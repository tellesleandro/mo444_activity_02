import os
import sys
import numpy as np
from pdb import set_trace as bp
from matplotlib import pyplot as plt

source_path = sys.argv[1]

i = 1
for camera in os.listdir(source_path):

    histogram_dir = source_path + camera + '/'

    hist_b = np.load(histogram_dir + 'fingerprint_hist_b.npy')
    hist_g = np.load(histogram_dir + 'fingerprint_hist_g.npy')
    hist_r = np.load(histogram_dir + 'fingerprint_hist_r.npy')

    plt.subplot(10,3,i)
    plt.plot(range(256), hist_b)
    i += 1

    plt.subplot(10,3,i)
    plt.plot(range(256), hist_g)
    i += 1

    plt.subplot(10,3,i)
    plt.plot(range(256), hist_r)
    i += 1

plt.show()
