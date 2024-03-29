#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import dtw.dtw
from typing import Tuple

"""
This script creates a bridge plot with dynamic time warping.
I assume it is called a bridge plot because it looks like a bridge (kind of).
"""

samples: int = 200
x: np.ndarray = np.linspace(0, 2*np.pi, samples + 1)

sin_offset: int = 0
sin_x: np.ndarray = np.sin(x) + sin_offset
sin_x: np.ndarray = sin_x.reshape(-1, 1)
cos_x: np.ndarray = np.cos(x).reshape(-1, 1)
result: Tuple[float, float, np.ndarray, Tuple[np.ndarray, np.ndarray]] = \
    dtw.dtw(sin_x, cos_x, dist=lambda a, b: np.linalg.norm(a - b, ord=1))
dist: float = result[0]
cost: float = result[1]
acc: np.ndarray = result[2]
path: Tuple[np.ndarray, np.ndarray] = result[3]

# correct
sin_0_2pi_scale: np.ndarray = np.array([q / (samples / (2 * np.pi)) for q in path[0]])
cos_0_2pi_scale: np.ndarray = np.array([q / (samples / (2 * np.pi)) for q in path[1]])

# Choose this incorrect scaling to see some abstract art!!!
# sin_0_2pi_scale = np.array([q / (np.pi / 100.0) for q in path[0]])
# cos_0_2pi_scale = np.array([q / (np.pi / 100.0) for q in path[1]])

sin_x_mapping = np.sin(sin_0_2pi_scale) + sin_offset
cos_x_mapping = np.cos(cos_0_2pi_scale)
plt.plot(x, sin_x, linewidth=3, c='red')
plt.plot(x, cos_x, linewidth=3, c='blue')

sin_patch: mpatches.Patch = mpatches.Patch(color='red', label='Sin(x)')
cos_patch: mpatches.Patch = mpatches.Patch(color='blue', label='Cos(x)')
patches = [sin_patch, cos_patch]
plt.legend(handles=patches, loc='best', fontsize='x-large')

# this for loop plots all of the lines associated with the warping.
for i in range(0, len(sin_0_2pi_scale)):
    plt.plot([sin_0_2pi_scale[i], cos_0_2pi_scale[i]], [sin_x_mapping[i], cos_x_mapping[i]], 'k-')

plt.xlim((min(x), max(x)))

plt.title("Sin(x) vs. Cos(x) bridge plot", fontsize=20)
plt.xlabel("x", fontsize=14)
plt.ylabel("y", fontsize=14)


plt.show()
