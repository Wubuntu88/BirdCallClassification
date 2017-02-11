#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import dtw as dtw
samples = 200
x = np.linspace(0, 2*np.pi, samples + 1)

sin_offset = 2
sin_x = np.sin(x) + sin_offset
sin_x = sin_x.reshape(-1, 1)
cos_x = np.cos(x).reshape(-1, 1)
dist, cost, acc, path = dtw.dtw(sin_x, cos_x, dist=lambda a, b: np.linalg.norm(a - b, ord=1))

# correct
sin_0_2pi_scale = np.array([q / (samples / (2 * np.pi)) for q in path[0]])
cos_0_2pi_scale = np.array([q / (samples / (2 * np.pi)) for q in path[1]])

# Choose this incorrect scaling to see some abstract art!!!
# sin_0_2pi_scale = np.array([q / (np.pi / 100.0) for q in path[0]])
# cos_0_2pi_scale = np.array([q / (np.pi / 100.0) for q in path[1]])

sin_x_mapping = np.sin(sin_0_2pi_scale) + sin_offset
cos_x_mapping = np.cos(cos_0_2pi_scale)
plt.plot(x, sin_x)
plt.plot(x, cos_x)

# this for loop plots all of the lines associated with the warping.
for i in range(0, len(sin_0_2pi_scale)):
    plt.plot([sin_0_2pi_scale[i], cos_0_2pi_scale[i]], [sin_x_mapping[i], cos_x_mapping[i]], 'k-')

plt.show()
