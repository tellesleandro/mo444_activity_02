import os
import sys
import numpy as np
from PIL import Image
from imutils import paths
from pdb import set_trace as bp
from skimage.feature import local_binary_pattern

source_path = sys.argv[1]
target_path = sys.argv[2]

for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = Image.open(image_path)
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)

    print('Loading matrix of pixels from image')
    image_pixels = np.array(image)

    print('Calculating the 3-channel LBP')
    lbp_r = local_binary_pattern(image_pixels[:,:,0], 8, 1)
    lbp_g = local_binary_pattern(image_pixels[:,:,1], 8, 1)
    lbp_b = local_binary_pattern(image_pixels[:,:,2], 8, 1)

    print('Extracting 3-channel feature vector from image')
    (hist_r, _) = np.histogram(lbp_r, bins = 256)
    (hist_g, _) = np.histogram(lbp_g, bins = 256)
    (hist_b, _) = np.histogram(lbp_b, bins = 256)
    hist = np.concatenate((hist_r, hist_g, hist_b))

    print('Saving histograms')
    np.save(target_path + '/' + base + '_r', hist_r)
    np.save(target_path + '/' + base + '_g', hist_g)
    np.save(target_path + '/' + base + '_b', hist_b)
    np.save(target_path + '/' + base, hist)

    # print('Loading histograms')
    # hist_r = np.load(target_path + '/' + base + '_r.npy')
    # hist_g = np.load(target_path + '/' + base + '_g.npy')
    # hist_b = np.load(target_path + '/' + base + '_b.npy')
    #
    # print('Saving histograms')
    # hist = np.concatenate((hist_r, hist_g, hist_b))
    # np.save(target_path + '/' + base, hist)

    print()
