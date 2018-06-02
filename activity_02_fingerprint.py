import cv2
import sys
from scipy import signal
from imutils import paths
import numpy as np
from pdb import set_trace as bp

source_path = sys.argv[1]
target_path = sys.argv[2]

numerators = []
denominators = []

for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = cv2.imread(image_path)

    print('Calculating filter')
    image_denoised = signal.wiener(image)
    filter = image - image_denoised
    w_i = filter * image
    i_squared = image * image

    numerators.append(w_i)
    denominators.append(i_squared)

    print()

summed_numerators = np.sum(numerators, axis = 0)
summed_denominators = np.sum(denominators, axis = 0)
k = summed_numerators / summed_denominators

np.save(target_path + 'features', k)
