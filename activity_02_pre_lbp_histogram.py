import os
import sys
import cv2
import numpy as np
from imutils import paths
from pdb import set_trace as bp
from skimage.feature import local_binary_pattern

source_path = sys.argv[1]
target_path = sys.argv[2]

for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = cv2.imread(image_path)

    print('Calculating the 3-channel and mean LBP')
    lbp_b = local_binary_pattern(image[:,:,0], 8, 1).flatten()
    lbp_g = local_binary_pattern(image[:,:,1], 8, 1).flatten()
    lbp_r = local_binary_pattern(image[:,:,2], 8, 1).flatten()
    lbp_mean = np.mean((lbp_b, lbp_g, lbp_r), axis=0)

    print('Extracting histogram')
    (hist_b, _) = np.histogram(lbp_b, bins = 256)
    (hist_g, _) = np.histogram(lbp_g, bins = 256)
    (hist_r, _) = np.histogram(lbp_r, bins = 256)
    (hist_mean, _) = np.histogram(lbp_mean, bins = 256)

    hist = np.concatenate((hist_b, hist_g, hist_r))

    print('Saving histograms')
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    np.save(target_path + base + '_lbp_histogram_b', hist_b)
    np.save(target_path + base + '_lbp_histogram_g', hist_g)
    np.save(target_path + base + '_lbp_histogram_r', hist_r)
    np.save(target_path + base + '_lbp_histogram_mean', hist_mean)
    np.save(target_path + base + '_lbp_histogram_bgr', hist)

    print()
