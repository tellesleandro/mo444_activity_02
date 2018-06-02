import os
import pdb
import cv2
import pywt
import numpy as np
import csv
import pickle
from PIL import Image
from pdb import set_trace as bp
from imutils import paths
from sklearn.linear_model import LogisticRegression as LR
from texttable import Texttable

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
        print('[', i, ']:', 'Loading image ' + image_path)
        image = Image.open(image_path)

        print('[', i, ']:', 'Cropping 512x512 center from image')
        width, height = image.size
        new_width = 512
        new_height = 512
        left = (width - new_width)/2
        top = (height - new_height)/2
        image = image.crop((left, top, left + new_width, top + new_height))

        print('[', i, ']:', 'Calculating the DWT')
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs

        print('[', i, ']:', 'Extracting feature vector')
        (hist, _) = np.histogram((cH, cV, cD),
            bins = 'sqrt')

        print('[', i, ']:', 'Adding features to feature matrix')
        labels.append(image_path.split("/")[-2])
        data.append(hist.tolist())

        print()

        i += 1
        if i == 5000:
            break

    labels = np.array(labels)
    data = np.array(data)

    print('Saving features to', dataset_filename_x)
    np.save(dataset_filename_x, data)
    print('Saving target to', dataset_filename_y)
    np.save(dataset_filename_y, labels)

print('Feature matrix shape: ', data.shape)
print('Labels vector shape: ', labels.shape)
print()

logistic_regression_file = './logistic_regression.pkl'
if os.path.isfile(logistic_regression_file):
    print('Loading the already computed logistic regression in', logistic_regression_file)
    with open(logistic_regression_file, 'rb') as input:
        logistic = pickle.load(input)
else:
    print('Computing the logistic regression')
    logistic = LR(solver = 'sag', max_iter = 1000)
    result = logistic.fit(data, labels)
    print(result)
    print()

    print('Saving the logistic regression to', logistic_regression_file)
    with open(logistic_regression_file, 'wb') as output:
        pickle.dump(logistic, output, pickle.HIGHEST_PROTOCOL)

data = []
test_folder = './validation_test'

images_filename = './images.pkl'
training_set_filename = './training_set.npy'
if os.path.isfile(images_filename) and os.path.isfile(training_set_filename):
    print('Loading the already computed training set', training_set_filename)
    data = np.load(training_set_filename)
    with open(images_filename, 'rb') as input:
        images = pickle.load(input)
else:
    i = 0
    images = []
    for image_path in paths.list_images(test_folder):
        images.append(image_path)

        print('[', i, ']:', 'Loading image ' + image_path)
        image = Image.open(image_path)

        print('[', i, ']:', 'Cropping 512x512 center from image')
        width, height = image.size
        new_width = 512
        new_height = 512
        left = (width - new_width)/2
        top = (height - new_height)/2
        image = image.crop((left, top, left + new_width, top + new_height))

        print('[', i, ']:', 'Calculating the DWT')
        coeffs = pywt.dwt2(image, 'haar')
        cA, (cH, cV, cD) = coeffs

        print('[', i, ']:', 'Extracting feature vector')
        (hist, _) = np.histogram((cH, cV, cD),
            bins = 'sqrt')
        data.append(hist.tolist())

        print()

        i += 1
        if i == 1000:
            break

    print('Saving training_set to', training_set_filename)
    data = np.array(data)
    np.save(training_set_filename, data)
    with open(images_filename, 'wb') as output:
        pickle.dump(images, output, pickle.HIGHEST_PROTOCOL)

print('Predicting the cameras based on the images')
labels = logistic.predict(data)
predictions = dict(zip(images, labels))

print('Prediction list')
text_table = Texttable()
text_table.add_row(['Image', 'Predicted', 'Actual', 'Correct ?'])

right_classifier_counter = 0
for image in images:
    images_filename = image.split("/")[-1]
    actual_camera = image.split("/")[-2]
    predicted_camera = predictions[image]
    text_table.add_row([images_filename, predicted_camera, actual_camera,
                        predicted_camera == actual_camera])
    if actual_camera == predicted_camera: right_classifier_counter += 1

print(text_table.draw())
print('Accuracy:', right_classifier_counter / len(images))

submission_file = './submission.csv'
print("Saving Kaggle's submission file")
with open(submission_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['fname', 'camera'])
    for prediction in predictions:
        fname = prediction.split('/')[-1]
        camera = predictions[prediction]
        writer.writerow([fname, camera])
