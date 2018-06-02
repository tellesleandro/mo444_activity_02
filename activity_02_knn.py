import os
import sys
import pickle
import numpy as np
from pdb import set_trace as bp
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

training_path = 'validation_train_lbp_histograms/'
test_path = 'validation_test_lbp_histograms/'

saved_model_filename = 'knn_model.sav'

if os.path.isfile(saved_model_filename):

    print('Loading saved KNN model')
    knn = pickle.load(open(saved_model_filename, 'rb'))

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

    le = preprocessing.LabelEncoder()
    le.fit(np.unique(training_labels))
    training_labels_encoded = le.transform(training_labels)

    print('Training the KNN classifier')
    knn = KNeighborsClassifier(algorithm = 'ball_tree')
    knn.fit(training_data, training_labels_encoded)

    print('Saving KNN model')
    pickle.dump(knn, open(saved_model_filename, 'wb'))

test_data = []
test_labels = []

for camera in os.listdir(test_path):

    for histogram_file in os.listdir(test_path + camera):

        histogram_full_file = test_path + camera + '/' + histogram_file

        if histogram_full_file.endswith('_lbp_histogram_mean.npy'):
            print('Loading test data', histogram_full_file)
            histogram = np.load(histogram_full_file).tolist()
            test_data.append(histogram)
            test_labels.append(camera)

test_data = np.array(test_data)
test_labels = np.array(test_labels)

le = preprocessing.LabelEncoder()
le.fit(np.unique(test_labels))
test_labels_encoded = le.transform(test_labels)

print('Predicting labels')
predicted_labels = knn.predict(test_data)

print('Calculating accuracy')
accuracy = knn.score(test_data, test_labels_encoded)
print(accuracy)

print('Creating a confusion matrix')
cm = confusion_matrix(test_labels_encoded, predicted_labels, labels = test_labels)
print(cm)

print(test_labels_encoded)
