#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import dtw.dtw
import librosa

file_path_1 = "../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/nips4b_birds_trainfile002.wav"
file_path_2 = "../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/nips4b_birds_trainfile093.wav"
y_1, sampling_rate_1 = librosa.load(file_path_1)
y_2, sampling_rate_2 = librosa.load(file_path_2)

mfcc_1 = librosa.feature.mfcc(y_1, sampling_rate_1, n_mfcc=13).T
mfcc_2 = librosa.feature.mfcc(y_2, sampling_rate_2, n_mfcc=13).T

dist, cost, acc, path = dtw.dtw(mfcc_1, mfcc_2, dist=lambda a, b: np.linalg.norm(a - b, ord=1))
print("dist: ", dist)

plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))
plt.show()
