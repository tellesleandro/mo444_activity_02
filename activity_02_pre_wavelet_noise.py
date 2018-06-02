import os
import sys
import cv2
from imutils import paths
from pdb import set_trace as bp

source_path = sys.argv[1]
target_path = sys.argv[2]

i = 0
for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = cv2.imread(image_path)

    print('Calculating image noise using NL Means')
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    # denoised_image = denoise_wavelet(image, multichannel = True)
    noise = image - denoised_image

    print('Saving oringal image and noise')
    filename = os.path.basename(image_path)
    cv2.imwrite(target_path + 'original_' + filename, image)
    cv2.imwrite(target_path + 'denoised_' + filename, denoised_image)
    cv2.imwrite(target_path + 'noise_' + filename, noise)

    print()
