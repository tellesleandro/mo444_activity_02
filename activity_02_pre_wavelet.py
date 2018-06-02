import os
import sys
import numpy as np
from PIL import Image
from imutils import paths
from pdb import set_trace as bp
from skimage.restoration import denoise_wavelet

source_path = sys.argv[1]
target_path = sys.argv[2]

for image_path in paths.list_images(source_path):

    print('Loading image', image_path)
    image = Image.open(image_path)
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)

    print('Calculating image noise using DWT')
    denoised_image = denoise_wavelet(image, multichannel = True)
    if not np.isnan(np.sum(denoised_image)):
        # There is noise in the picture
        noise = image - denoised_image

        print('Extracting 3-channel feature vector from noise')
        (hist_r, _) = np.histogram(noise[:,:,0], bins = 256)
        (hist_g, _) = np.histogram(noise[:,:,1], bins = 256)
        (hist_b, _) = np.histogram(noise[:,:,2], bins = 256)

    else:
        # There is no noise in the picture
        print('Generating zeroed 3-channel feature vector')
        hist_r = np.zeros(256)
        hist_g = np.zeros(256)
        hist_b = np.zeros(256)

    print('Saving histograms')
    hist = np.concatenate((hist_r, hist_g, hist_b))
    np.save(target_path + '/' + base + '_r', hist_r)
    np.save(target_path + '/' + base + '_g', hist_g)
    np.save(target_path + '/' + base + '_b', hist_b)
    np.save(target_path + '/' + base, hist)

    print()
    
