#!/usr/bin/python
import librosa
from librosa import display, amplitude_to_db
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from typing import Tuple

'''
This script loads each training data audio file and creates a mel spectrogram of it.
Each mel spectrogram is saved in the directory out_melspectrograms.
A spectrogram shows the time on the x axis and the frequency on the y axis.
Frequencies areas with darker colors represent frequencies that with higher amplitudes
(i.e. frequencies that are louder in the audio file).
This script is quite CPU and I/O intensive.
'''

relative_path_to_data_directory = '../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train'
file_names = os.listdir(relative_path_to_data_directory)

i = 1
for file_name in file_names:
    path_to_file = relative_path_to_data_directory + "/" + file_name
    audio_time_series_sampling_rate_tuple: Tuple[np.ndarray, int] = librosa.load(path_to_file)
    y, sr = audio_time_series_sampling_rate_tuple
    # Passing through arguments to the Mel filters
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)  # amplitude spectrogram
    dB_scaled_spectrogram = librosa.amplitude_to_db(
        spectrogram,
        ref=np.max)

    fig, ax = plt.subplots()

    img = librosa.display.specshow(
        dB_scaled_spectrogram,
        x_axis='time', y_axis='mel',
        sr=sr,
        fmax=8000, ax=ax)

    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    ax.set(title='Mel-frequency spectrogram')

    plt.savefig("../out_melspectrograms/" + "melspec" + '{0:03d}'.format(i) + ".png")
    plt.clf()
    i += 1
