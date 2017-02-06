#!/usr/bin/python
import librosa
import numpy as np
import matplotlib.pyplot as plt

file_path = "../../NIPS4B_BIRD_CHALLENGE_TRAIN_TEST_WAV/train/nips4b_birds_trainfile012.wav"

y, sr = librosa.load(file_path)

freq_min = 500
freq_max = 8000

# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=freq_min, fmax=freq_max)


librosa.display.specshow(librosa.logamplitude(S,
                                              ref_power=np.max),
                         y_axis='mel', fmin=freq_min, fmax=freq_max,
                         x_axis='time')

plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
