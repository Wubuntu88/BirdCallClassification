#!/usr/bin/python
import librosa
import numpy as np
import matplotlib.pyplot as plt
f_name = "trainfile016_CettisWarbler03"

file_path = "../zShortBirdRecordings/" + f_name + ".wav"

y, sr = librosa.load(file_path)

freq_min = 500
freq_max = 8000

# Passing through arguments to the Mel filters
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmin=freq_min, fmax=freq_max)


librosa.display.specshow(librosa.logamplitude(S,
                                              ref_power=np.max),
                         y_axis='mel', fmin=freq_min, fmax=freq_max,
                         x_axis='time')

plt.colorbar(format='%+2.0f dBFS')
plt.title('Mel spectrogram')
plt.tight_layout()
plt.show()
#plt.savefig(filename="../zShortMelSpectrograms/" + f_name + ".png")
