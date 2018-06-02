import os
import numpy as np
from pdb import set_trace as bp
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

print('Loading training data')
training_data = np.load('training_features_lbp_histogram.npy')
training_labels = np.load('training_labels_lbp_histogram.npy')
le = preprocessing.LabelEncoder()
le.fit(np.unique(training_labels))
training_labels_encoded = le.transform(training_labels)

print('Loading test data')
test_data = np.load('test_features_lbp_histogram.npy')
test_labels = np.load('test_labels_lbp_histogram.npy')
le = preprocessing.LabelEncoder()
le.fit(np.unique(test_labels))
test_labels_encoded = le.transform(test_labels)

print('Training the decision tree classifier')
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(training_data, training_labels_encoded)
dtree_predictions = dtree_model.predict(test_data)

print('Creating the confusion matrix')
cm = confusion_matrix(test_labels_encoded, dtree_predictions)
print(cm)
