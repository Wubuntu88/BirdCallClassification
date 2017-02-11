#!/usr/bin/python
import matplotlib.pyplot as plt
import dtw as dtw
import numpy as np
from matplotlib.ticker import FormatStrFormatter

x = np.linspace(0, 2*np.pi, 201)
sin_x = np.sin(x).reshape(-1, 1)
cos_x = np.cos(x).reshape(-1, 1)

dist, cost, acc, path = dtw.dtw(sin_x, cos_x, dist=lambda x, y: np.linalg.norm(x - y, ord=1))

# definitions for the axes
left, width = 0.12, 0.60
bottom, height = 0.08, 0.60
bottom_h = 0.16 + width
left_h = left + 0.27
rect_plot = [left_h, bottom, width, height]
rect_x = [left_h, bottom_h, width, 0.2]
rect_y = [left, bottom, 0.2, height]


# start with a rectangular Figure
fig = plt.figure(2, figsize=(8, 8))
fig.suptitle("sin(x) Vs. cos(x) Dynamic Time warping", fontsize=14)

axplot = plt.axes(rect_plot)
axx = plt.axes(rect_x)
axy = plt.axes(rect_y)

# Plot the matrix
axplot.pcolor(acc.T, cmap='nipy_spectral')
axplot.plot(path[0], path[1], 'w')


axplot.set_xlim((0, len(x)))
axplot.set_ylim((0, len(sin_x)))
axplot.tick_params(axis='both', which='major', labelsize=12)

# Plot time serie horizontal
axx.plot(sin_x, '.', color='k')
axx.tick_params(axis='both', which='major', labelsize=12)
xloc = plt.MaxNLocator(4)
x2Formatter = FormatStrFormatter('%d')
axx.yaxis.set_major_locator(xloc)
axx.yaxis.set_major_formatter(x2Formatter)

# Plot time serie vertical
axy.plot(np.cos(x), (100.0 / np.pi) * x, '.', color='k')
axy.invert_xaxis()
yloc = plt.MaxNLocator(4)
xFormatter = FormatStrFormatter('%d')
axy.xaxis.set_major_locator(yloc)
axy.xaxis.set_major_formatter(xFormatter)
axy.tick_params(axis='both', which='major', labelsize=18)

#Limits
axx.set_xlim(axplot.get_xlim())

axy.set_ylim(axplot.get_ylim())
plt.xlabel("cos(x)")
plt.ylabel("x")

plt.show()
