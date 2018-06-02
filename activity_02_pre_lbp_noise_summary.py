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
    noise = np.stack((lbp_b, lbp_g, lbp_r), axis=2)

    noises_and_images.append([noise, image])

    print()

print('Summarizing noises and images')
noises_and_images = np.array(noises_and_images)

print('Calculating noise fingerprint')
noise_image = noises_and_images[:,0] * noises_and_images[:,1]
summed_noise_image = np.sum(noise_image, axis=0)
image_squared = noises_and_images[:,1] * noises_and_images[:,1]
summed_image_squared = np.sum(image_squared, axis=0)
fingerprint = summed_noise_image / summed_image_squared

print('Saving noise fingerprint')
np.save(target_path + 'fingerprint_matrix', fingerprint)
(hist_b, _) = np.histogram(fingerprint[:, :, 0], bins = 256)
(hist_g, _) = np.histogram(fingerprint[:, :, 1], bins = 256)
(hist_r, _) = np.histogram(fingerprint[:, :, 2], bins = 256)
np.save(target_path + 'fingerprint_hist_b', hist_b)
np.save(target_path + 'fingerprint_hist_g', hist_g)
np.save(target_path + 'fingerprint_hist_r', hist_r)
