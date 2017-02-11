#!/usr/bin/python
import librosa
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches

"""
This script will create mfccs from a file, and then plot them with 'time' on the x axis
and the value of the mfcc on the y axis.  Different mfccs are shown in different colors,
that way we can show multiple dimensions on the same plot.
"""

file_path = "../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/nips4b_birds_trainfile015.wav"

y, sr = librosa.load(file_path)

freq_min = 500
freq_max = 8000
num_mfccs = 13

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfccs, fmin=freq_min, fmax=freq_max)
num_rows, num_cols = mfccs.shape
t = range(0, num_cols)

colors = cm.rainbow(np.linspace(0, 1, num_rows))

patches = []
for i in range(0, num_mfccs):
    patch = mpatches.Patch(color=colors[i], label='mfcc #{0:01d}'.format(i))
    patches.append(patch)

plt.legend(handles=patches, loc='center right')


for i in range(1, num_rows):
    plt.plot(t, mfccs[i], color=colors[i], linewidth=3)

plt.show()
