#!/usr/bin/python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import MFCC_Train_Test_Loader.loader as loader
import label_writer.writer as writer

train_features_files_path = "../../aggregate_mfcc_data/train/agg_mfccs.txt"
train_labels_file_path = "../../NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/HolyLabelMatrix.csv"
test_features_file_path = "../../aggregate_mfcc_data/test/agg_mfccs_test.txt"

features_matrix, labels_matrix, test_matrix = \
    loader.load_mfcc_train_features_labels_and_test_labels(train_features_file_path=train_features_files_path,
                                                           train_labels_file_path=train_labels_file_path,
                                                           test_features_file_path=test_features_file_path)

clf = DecisionTreeClassifier()
clf.fit(features_matrix, labels_matrix)

result = clf.predict(test_matrix)

file_path = "z_label_predictions/test.csv"
writer.write_label_predictions_to_file(relative_file_path=file_path, labels=result)

