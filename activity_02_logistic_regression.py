import os
import csv
import pickle
import numpy as np
from texttable import Texttable
from pdb import set_trace as bp
from sklearn.linear_model import LogisticRegression as LR

training_path = 'validation_train_lbp_histograms/'
test_path = 'test_lbp_histograms/'

saved_model_filename = 'logistic_regression_model.sav'
saved_result_filename = 'logistic_regression_result.sav'

if os.path.isfile(saved_model_filename) and os.path.isfile(saved_result_filename):

    print('Loading saved logistic regression model')
    logistic = pickle.load(open(saved_model_filename, 'rb'))
    result = pickle.load(open(saved_result_filename, 'rb'))

else:

    training_data = []
    training_labels = []

    for camera in os.listdir(training_path):

        for histogram_file in os.listdir(training_path + camera):

            histogram_full_file = training_path + camera + '/' + histogram_file

            if histogram_full_file.endswith('_lbp_histogram_mean.npy'):

                print('Loading training data', histogram_full_file)
                histogram = np.load(histogram_full_file).tolist()
                training_data.append(histogram)
                training_labels.append(camera)

    training_data = np.array(training_data)
    training_labels = np.array(training_labels)

    print('Computing logistic regression')
    logistic = LR(solver = 'lbfgs', multi_class='multinomial', max_iter = 5000, verbose = True)
    result = logistic.fit(training_data, training_labels)

    print('Saving logistic regression model')
    pickle.dump(logistic, open(saved_model_filename, 'wb'))
    pickle.dump(result, open(saved_result_filename, 'wb'))

images = []
test_data = []
test_labels = []

# for camera in os.listdir(test_path):

# for histogram_file in os.listdir(test_path + camera):
for histogram_file in os.listdir(test_path):

    # histogram_full_file = test_path + camera + '/' + histogram_file
    histogram_full_file = test_path + histogram_file

    if histogram_full_file.endswith('_lbp_histogram_mean.npy'):

        filename = os.path.basename(histogram_full_file)
        images.append(filename.replace('_lbp_histogram_mean.npy', ''))

        print('Loading test data', histogram_full_file)
        histogram = np.load(histogram_full_file).tolist()
        test_data.append(histogram)
        # test_labels.append(camera)

test_data = np.array(test_data)
# test_labels = np.array(test_labels)

print('Predicting labels')
predicted_labels = logistic.predict(test_data)
predictions = dict(zip(images, predicted_labels))

# accuracy = {}
# for i, predicted_label in enumerate(predicted_labels):
#     if test_labels[i] not in accuracy:
#         accuracy[test_labels[i]] = [0, 0]
#     if predicted_label == test_labels[i]:
#         accuracy[test_labels[i]][0] += 1
#     else:
#         accuracy[test_labels[i]][1] += 1
#
# total_accuracy = 0
# for key, value in accuracy.items():
#     camera_accuracy = value[0] / (value[0] + value[1])
#     total_accuracy += camera_accuracy
#     print('Accuracy [', key, ']', camera_accuracy)
#
# print('Mean accuracy', total_accuracy / len(accuracy))

bp()

submission_file = './submission.csv'
print("Saving Kaggle's submission file")
with open(submission_file, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['fname', 'camera'])
    for prediction in predictions:
        fname = prediction + '.tif'
        camera = predictions[prediction]
        writer.writerow([fname, camera])
