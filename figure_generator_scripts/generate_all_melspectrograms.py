#!/usr/bin/python
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os

'''
This script loads each training data audio file and creates a mel spectrogram of it.
Each mel spectrogram is saved in the directory z_melspectrograms.  The z is there to
push it to the bottom of the directory list.
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
    y, sr = librosa.load(path_to_file)
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)

    librosa.display.specshow(librosa.logamplitude(S,
                                                  ref_power=np.max),
                             y_axis='mel', fmax=8000,
                             x_axis='time')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()

    plt.savefig("../z_melspectrograms/" + "melspec" + '{0:03d}'.format(i) + ".png")
    plt.clf()
    i += 1
