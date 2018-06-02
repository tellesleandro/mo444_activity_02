import os
import sys
import glob
import numpy as np
from pdb import set_trace as bp

source_path = sys.argv[1]
target_path = sys.argv[2]

camera_data = []
camera_labels = []

general_data = []
general_labels = []

for camera in os.listdir(source_path):

    for histogram_file in os.listdir(source_path + camera):

        if histogram_file.endswith('_r.npy') \
            or histogram_file.endswith('_g.npy') \
            or histogram_file.endswith('_b.npy'):
            continue

        source_file = source_path + camera + '/' + histogram_file

        print('Loading histogram from', source_file)
        camera_data.append(np.load(source_file))
        camera_labels.append(camera)

    print('Writing feature matrix for', camera)
    target_file = target_path + '/' + camera + '/features'
    np_camera_data = np.array(camera_data)
    np.save(target_file, np_camera_data)

    if len(general_data) == 0:
        general_data = np_camera_data
        general_labels = np.array([camera] * len(np_camera_data))
    else:
        general_data = np.concatenate((general_data, np_camera_data))
        general_labels = np.concatenate((general_labels, [camera] * len(np_camera_data)))

    print()

print('Writing feature matrix for all cameras')
features_target_file = target_path + '/features'
np.save(features_target_file, general_data)
labels_target_file = target_path + '/labels'
np.save(labels_target_file, general_labels)
