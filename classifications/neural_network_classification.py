#!/usr/bin/python
from sklearn.neural_network import MLPClassifier
import MFCC_Train_Test_Loader.loader as loader
import label_writer.writer as writer
from typing import List

train_features_files_path = "../aggregate_mfcc_data/train/agg_mfccs_train.txt"
train_labels_file_path = "../NIPS4B_BIRD_CHALLENGE_TRAIN_LABELS/LabelMatrix.csv"
test_features_file_path = "../aggregate_mfcc_data/test/agg_mfccs_test.txt"

features_matrix, labels_matrix, test_matrix = \
    loader.load_mfcc_train_features_labels_and_test_labels(train_features_file_path=train_features_files_path,
                                                           train_labels_file_path=train_labels_file_path,
                                                           test_features_file_path=test_features_file_path)
alpha: float = 0.1
max_iter: int = 500
solvers: List[str] = ["lbfgs", "sgd", "adam"]
solver: str = "adam"
clf = MLPClassifier(solver='adam', alpha=alpha, max_iter=max_iter)
clf.fit(features_matrix, labels_matrix)

result = clf.predict_proba(test_matrix)

total = 0
for k in range(0, len(result)):
    for i in range(0, len(result[0])):
        if result[k][i] > .3:
            print("k: ", k, ", i", i, "result: ", result[k][i])
            total += 1

print("total records with prediction prob > .3: ", total)

file_path = "../label_predictions/test_ann_alpha={}_iter={}_solver={}.csv".format(alpha, max_iter, solver)
writer.write_label_predictions_to_file(relative_file_path=file_path, labels=result)
