#!/usr/bin/python
import numpy as np
import matplotlib.pyplot as plt
from dtw import dtw
x = np.linspace(-np.pi, np.pi, 201)
# sin_x = np.sin(x + np.pi / 2).reshape(-1, 1)
sin_x = np.sin(x).reshape(-1, 1)
cos_x = np.cos(x).reshape(-1, 1)
# x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
# y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)
dist, cost, acc, path = dtw(sin_x, cos_x, dist=lambda a, b: np.linalg.norm(a - b, ord=1))
plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.xlim((-0.5, acc.shape[0]-0.5))
plt.ylim((-0.5, acc.shape[1]-0.5))
plt.show()
