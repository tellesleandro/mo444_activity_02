import os
import cv2
import sys
import glob
from imutils import paths
# import numpy as np
from pdb import set_trace as bp
# from skimage.restoration import denoise_wavelet

source_path = sys.argv[1]
# target_path = sys.argv[2]

noises_and_images = []

i = 0
for image_path in glob.glob(source_path + 'original_*')

    original_file = image_path
    noise_file = image_path.replace('original_', 'noise_')

    print('Loading noise and image', image_path)
    noise = cv2.imread(noise_file)
    image = cv2.imread(original_file)

    print('Saving noise and image')
    noises_and_images.append([noise, image])

    print()

    i = i+1
    if i == 2:
        break

noises_and_images = np.array(noises_and_images)

print('Calculating noise fingerprint')
noise_image = noises_and_images[:,0] * noises_and_images[:,1]
summed_noise_image = np.sum(noise_image, axis=0)
image_squared = noises_and_images[:,1] * noises_and_images[:,1]
summed_image_squared = np.sum(image_squared, axis=0)
fingerprint = summed_image_noise / summed_image_squared
