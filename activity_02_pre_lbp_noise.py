import os
import sys
import cv2
import numpy as np
from imutils import paths
from pdb import set_trace as bp
from skimage.feature import local_binary_pattern

source_path = sys.argv[1]
target_path = sys.argv[2]

noises_and_images = []

i = 0
for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = cv2.imread(image_path)

    print('Calculating the 3-channel LBP')
    lbp_b = local_binary_pattern(image[:,:,0], 8, 1)
    lbp_g = local_binary_pattern(image[:,:,1], 8, 1)
    lbp_r = local_binary_pattern(image[:,:,2], 8, 1)

    print('Extracting 3-channel feature vector from image')
    (hist_b, _) = np.histogram(lbp_b, bins = 256)
    (hist_g, _) = np.histogram(lbp_g, bins = 256)
    (hist_r, _) = np.histogram(lbp_r, bins = 256)
    hist = np.concatenate((hist_b, hist_g, hist_r))

    print('Saving 3-channel LBP')
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    np.save(target_path + base + '_bgr_histogram', hist)

    print()
