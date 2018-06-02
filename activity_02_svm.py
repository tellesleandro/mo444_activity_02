import sys
import numpy as np
from pdb import set_trace as bp
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

training_path = sys.argv[1]
test_path = sys.argv[2]

print('Loading features and labels from the training set')
features_file = training_path + '/features.npy'
labels_file = training_path + '/labels.npy'
training_features = np.load(features_file)
training_labels = np.load(labels_file)
le = preprocessing.LabelEncoder()
le.fit(np.unique(training_labels))
training_labels_encoded = le.transform(training_labels)

print('Loading features and labels from the test set')
features_file = test_path + '/features.npy'
labels_file = test_path + '/labels.npy'
test_features = np.load(features_file)
test_labels = np.load(labels_file)
le = preprocessing.LabelEncoder()
le.fit(np.unique(test_labels))
test_labels_encoded = le.transform(test_labels)

print('Training the linear SVM classifier')
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(training_features, training_labels_encoded)

print('Predicting camera models')
svm_predictions = svm_model_linear.predict(test_features)

print('Calculating accuracy')
accuracy = svm_model_linear.score(test_features, test_labels_encoded)

print('Creating a confusion matrix')
cm = confusion_matrix(test_labels_encoded, dtree_predictions)
print(cm)
