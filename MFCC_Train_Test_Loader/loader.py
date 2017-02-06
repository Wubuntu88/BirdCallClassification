import numpy as np


def load_mfcc_train_features_labels_and_test_labels(train_features_file_path,
                                                    train_labels_file_path,
                                                    test_features_file_path):
    train_features_matrix = np.loadtxt(train_features_file_path)
    train_labels_matrix = np.loadtxt(train_labels_file_path, dtype=int, delimiter=",")
    test_labels_matrix = np.loadtxt(test_features_file_path)
    return train_features_matrix, train_labels_matrix, test_labels_matrix
