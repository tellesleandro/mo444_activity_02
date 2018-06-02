import os
import pdb
import cv2
import numpy as np
import csv
import pickle
import scipy.misc
import datetime
import matplotlib.pyplot as plt
from PIL import Image
from pdb import set_trace as bp
from imutils import paths
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression as LR
from texttable import Texttable
from skimage.restoration import denoise_wavelet
from multiprocessing import Process

def log(*message):
    print(datetime.datetime.now().strftime("%d.%b %Y %H:%M:%S"), *message)

print()

data = []
labels = []
training_folder = './validation_train'

dataset_filename_x = './dataset_x.npy'
dataset_filename_y = './dataset_y.npy'
if os.path.isfile(dataset_filename_x) and os.path.isfile(dataset_filename_y):
    data = np.load(dataset_filename_x)
    labels = np.load(dataset_filename_y)
else:
    i = 0
    for image_path in paths.list_images(training_folder):

        log('[', i, ']:', 'Loading image ' + image_path)

        original_image = Image.open(image_path)
        width, height = original_image.size
        new_width = 512
        new_height = 512

        # Crop 5 regions from the image: 4 edges and center
        for j in range(5):

            if j==0:
                # Crop the center
                left = (width - new_width) / 2
                top = (height - new_height)/2
                right = left + new_width
                bottom = top + new_height
                log('[', i, ']:', 'Cropping 512x512 center from image')
            elif j==1:
                # Crop the top-left edge
                left = 0
                top = 0
                right = left + new_width
                bottom = top + new_height
                log('[', i, ']:', 'Cropping 512x512 top-left from image')
            elif j==2:
                # Crop the top-right edge
                left = width - new_width
                top = 0
                right = left + new_width
                bottom = top + new_height
                log('[', i, ']:', 'Cropping 512x512 top-right from image')
            elif j==3:
                # Crop the bottom-left edge
                left = 0
                top = height - new_height
                right = left + new_width
                bottom = top + new_height
                log('[', i, ']:', 'Cropping 512x512 bottom-left from image')
            elif j==4:
                # Crop the bottom-right edge
                left = width - new_width
                top = height - new_height
                right = left + new_width
                bottom = top + new_height
                log('[', i, ']:', 'Cropping 512x512 bottom-right from image')

            image = original_image.crop((left, top, right, bottom))

            log('[', i, ']:', 'Loading matrix of pixels from image')
            image_pixels = np.array(image)

            log('[', i, ']:', 'Calculating the 3-channel LBP')
            lbp_r = local_binary_pattern(image_pixels[:,:,0], 8, 1)
            lbp_g = local_binary_pattern(image_pixels[:,:,1], 8, 1)
            lbp_b = local_binary_pattern(image_pixels[:,:,2], 8, 1)

            log('[', i, ']:', 'Extracting 3-channel feature vector from image')
            (hist_r_image, _) = np.histogram(lbp_r, bins = 256)
            (hist_g_image, _) = np.histogram(lbp_g, bins = 256)
            (hist_b_image, _) = np.histogram(lbp_b, bins = 256)

            log('[', i, ']:', 'Calculating image noise using DWT')
            denoised_image = denoise_wavelet(image, multichannel = True)
            if not np.isnan(np.sum(denoised_image)):
                # There is noise in the picture
                noise = image - denoised_image

                log('[', i, ']:', 'Extracting 3-channel feature vector from noise')
                (hist_r_noise, _) = np.histogram(noise[:,:,0], bins = 256)
                (hist_g_noise, _) = np.histogram(noise[:,:,1], bins = 256)
                (hist_b_noise, _) = np.histogram(noise[:,:,2], bins = 256)

            else:
                # There is no noise in the picture
                log('[', i, ']:', 'Generating 3-channel feature vector from noise')
                hist_r_noise = np.zeros(256)
                hist_g_noise = np.zeros(256)
                hist_b_noise = np.zeros(256)

            hist_r = np.multiply(hist_r_image, hist_r_noise)
            hist_g = np.multiply(hist_g_image, hist_g_noise)
            hist_b = np.multiply(hist_g_image, hist_b_noise)

            hist = np.concatenate((hist_r, hist_g, hist_b))
            plt.plot(range(768),hist)
            plt.show()

            bp()

            log('[', i, ']:', 'Adding features to feature matrix')
            labels.append(image_path.split("/")[-2])
            data.append(hist.tolist())

        print()

        i += 1
        if i == 5000:
            break

    labels = np.array(labels)
    data = np.array(data)

    log('Saving features to', dataset_filename_x)
    np.save(dataset_filename_x, data)
    log('Saving target to', dataset_filename_y)
    np.save(dataset_filename_y, labels)

log('Feature matrix shape: ', data.shape)
log('Labels vector shape: ', labels.shape)
print()

logistic_regression_file = './logistic_regression.pkl'
if os.path.isfile(logistic_regression_file):
    log('Loading the already computed logistic regression in', logistic_regression_file)
    with open(logistic_regression_file, 'rb') as input:
        logistic = pickle.load(input)
else:
    log('Computing the logistic regression')
    logistic = LR(solver = 'sag', max_iter = 10000)
    result = logistic.fit(data, labels)
    log(result)
    print()

    log('Saving the logistic regression to', logistic_regression_file)
    with open(logistic_regression_file, 'wb') as output:
        pickle.dump(logistic, output, pickle.HIGHEST_PROTOCOL)

data = []
test_folder = './test'

images_filename = './images.pkl'
training_set_filename = './training_set.npy'
if os.path.isfile(images_filename) and os.path.isfile(training_set_filename):
    log('Loading the already computed training set', training_set_filename)
    data = np.load(training_set_filename)
    with open(images_filename, 'rb') as input:
        images = pickle.load(input)
else:
    i = 0
    images = []
    for image_path in paths.list_images(test_folder):

        images.append(image_path)

        log('[', i, ']:', 'Loading image ' + image_path)
        original_image = Image.open(image_path)
        width, height = original_image.size
        new_width = 512
        new_height = 512

        if test_folder == './test':
            # There is no need to crop the test dataset
            image = original_image
        else:
            # Crop the center
            left = (width - new_width) / 2
            top = (height - new_height)/2
            right = left + new_width
            bottom = top + new_height
            log('[', i, ']:', 'Cropping 512x512 center from image')
            image = original_image.crop((left, top, right, bottom))

        log('[', i, ']:', 'Loading matrix of pixels from image')
        image_pixels = np.array(image)

        log('[', i, ']:', 'Calculating the 3-channel LBP')
        lbp_r = local_binary_pattern(image_pixels[:,:,0], 8, 1)
        lbp_g = local_binary_pattern(image_pixels[:,:,1], 8, 1)
        lbp_b = local_binary_pattern(image_pixels[:,:,2], 8, 1)

        log('[', i, ']:', 'Extracting 3-channel feature vector from image')
        (hist_r_image, _) = np.histogram(lbp_r, bins = 256)
        (hist_g_image, _) = np.histogram(lbp_g, bins = 256)
        (hist_b_image, _) = np.histogram(lbp_b, bins = 256)

        log('[', i, ']:', 'Calculating image noise using DWT')
        denoised_image = denoise_wavelet(image, multichannel = True)
        if not np.isnan(np.sum(denoised_image)):
            # There is noise in the picture
            noise = image - denoised_image

            log('[', i, ']:', 'Extracting 3-channel feature vector from noise')
            (hist_r_noise, _) = np.histogram(noise[:,:,0], bins = 256)
            (hist_g_noise, _) = np.histogram(noise[:,:,1], bins = 256)
            (hist_b_noise, _) = np.histogram(noise[:,:,2], bins = 256)

        else:
            # There is no noise in the picture
            log('[', i, ']:', 'Generating 3-channel feature vector from noise')
            hist_r_noise = np.zeros(256)
            hist_g_noise = np.zeros(256)
            hist_b_noise = np.zeros(256)

        hist = np.concatenate((hist_r_image, hist_g_image, hist_b_image,
                                hist_r_noise, hist_g_noise, hist_b_noise))

        log('[', i, ']:', 'Adding features to feature matrix')
        data.append(hist.tolist())

        print()

        i += 1
        if i == 5000:
            break

    log('Saving training_set to', training_set_filename)
    data = np.array(data)
    np.save(training_set_filename, data)
    with open(images_filename, 'wb') as output:
        pickle.dump(images, output, pickle.HIGHEST_PROTOCOL)

log('Predicting the cameras based on the images')
labels = logistic.predict(data)
predictions = dict(zip(images, labels))

log('Prediction list')
text_table = Texttable()
text_table.add_row(['Image', 'Predicted', 'Actual', 'Correct ?'])

if test_folder == './validation_test':
    right_classifier_counter = 0
    for image in images:
        images_filename = image.split("/")[-1]
        actual_camera = image.split("/")[-2]
        predicted_camera = predictions[image]
        text_table.add_row([images_filename, predicted_camera, actual_camera,
                            predicted_camera == actual_camera])
        if actual_camera == predicted_camera: right_classifier_counter += 1

    log(text_table.draw())
    log('Accuracy:', right_classifier_counter / len(images))

submission_file = './submission.csv'
log("Saving Kaggle's submission file")
with open(submission_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['fname', 'camera'])
    for prediction in predictions:
        fname = prediction.split('/')[-1]
        camera = predictions[prediction]
        writer.writerow([fname, camera])
