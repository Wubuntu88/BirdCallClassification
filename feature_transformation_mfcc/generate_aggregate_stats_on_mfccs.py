#!/usr/bin/python
import numpy as np
import os
import sys
train_or_test = ""
if len(sys.argv) == 2:
    if sys.argv[1] not in {"train", "test"}:
        print("The first argument must be train or test")
    else:
        train_or_test = sys.argv[1]
    print(train_or_test)
else:
    exit()


def mad(data, axis=None):
    return np.mean(np.absolute(data - np.mean(data, axis)), axis)


relative_path_to_data_directory = '../../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_MFCC/' + train_or_test
file_names = os.listdir(relative_path_to_data_directory)

output_file = open("../../aggregate_mfcc_data/" + train_or_test + "/agg_mfccs_" + train_or_test + ".txt", "w")
iteration = 1
for f_name in file_names:
    path_to_file = relative_path_to_data_directory + "/" + f_name
    matrix = np.loadtxt(path_to_file)

    aggregate_data = []

    # the cepstrum vector represents a given cepstrum over all of the time frames
    for cepstrum_vector in matrix:
        if f_name == "cepst_conc_cepst_nips4b_birds_trainfile060.txt":
            indices_of_NAN = np.where(np.isnan(cepstrum_vector))[0]  # index 0 gets array of indices
            last_index_of_NAN = -1
            last_index_of_INF = -1
            if len(indices_of_NAN) > 0:
                last_index_of_NAN = indices_of_NAN[-1]

            indices_of_INF = np.where(np.isinf(cepstrum_vector))[0]  # index 0 gets array of indices
            if len(indices_of_INF) > 0:
                last_index_of_INF = indices_of_INF[-1]

            index_to_cut_off_at = last_index_of_NAN if last_index_of_NAN > last_index_of_INF else last_index_of_INF
            cepstrum_vector = cepstrum_vector[index_to_cut_off_at + 1:]

        average_of_cepstrum = np.average(cepstrum_vector)
        median_of_cepstrum = np.median(cepstrum_vector)
        standard_deviation_of_cepstrum = np.std(cepstrum_vector)
        mean_abs_deviation = mad(data=cepstrum_vector)
        aggregate_data.extend((average_of_cepstrum, median_of_cepstrum, standard_deviation_of_cepstrum, mean_abs_deviation))
    to_write = "\t".join([str(x) for x in aggregate_data]) + "\n"
    output_file.write(to_write)
    print("iteration: ", iteration)
    iteration += 1

output_file.close()
