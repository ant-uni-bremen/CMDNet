#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 12:57:08 2021

@author: beck
"""
import numpy as np
import tikzplotlib as tplt
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology
from scipy.signal import find_peaks
# import random
# from mpl_toolkits.mplot3d import Axes3D

# Written for revision to show that concrete MAP is not convex
# New tikz plot version available with same name
# But still extract minima from this script


def detect_local_minima(arr):
    # https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array/3689710#3689710
    """
    Takes an array and detects the troughs using the local maximum filter.
    Returns a boolean mask of the troughs (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """
    # define an connected neighborhood
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#generate_binary_structure
    neighborhood = morphology.generate_binary_structure(len(arr.shape), 2)
    # apply the local minimum filter; all locations of minimum value
    # in their neighborhood are set to 1
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.filters.html#minimum_filter
    local_min = (filters.minimum_filter(arr, footprint=neighborhood) == arr)
    # local_min is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.
    #
    # we create the mask of the background
    background = (arr == 0)
    #
    # a little technicality: we must erode the background in order to
    # successfully subtract it from local_min, otherwise a line will
    # appear along the background border (artifact of the local minimum filter)
    # http://www.scipy.org/doc/api_docs/SciPy.ndimage.morphology.html#binary_erosion
    eroded_background = morphology.binary_erosion(
        background, structure=neighborhood, border_value=1)
    #
    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_min mask
    detected_minima = local_min ^ eroded_background
    return np.where(detected_minima)

# def fun(x, y):
#     return x**2 + y


def fun(x, y):
    tau = 1
    return (-1 * np.exp(x/tau) + 1 * np.exp(y/tau)) / (np.exp(x/tau) + np.exp(y/tau))


def fun2(x, y):
    tau = 0.1  # 0.1
    alpha = 0.2  # 0.5
    noise_var = 0.25  # 2 ** 2
    yr = -1  # 0.4
    # return (yr - (-1 * np.exp((x + np.log(alpha)) / tau) + 1 * np.exp((y + np.log(1 - alpha)) / tau)) / (np.exp((x + np.log(alpha)) / tau) + np.exp((y + np.log(1 - alpha)) / tau))) ** 2 + 2* noise_var * (np.exp(-x) + np.exp(-y) + x + y)
    return 1 / (2 * noise_var) * (yr - (-1 * np.exp((x + np.log(alpha)) / tau) + 1 * np.exp((y + np.log(1 - alpha)) / tau)) / (np.exp((x + np.log(alpha)) / tau) + np.exp((y + np.log(1 - alpha)) / tau))) ** 2 + (np.exp(-x) + np.exp(-y) + x + y)


region = [-2.0, 2.0, 0.01]
fig = plt.figure(1)
ax = fig.add_subplot(111, projection='3d')
x = y = np.arange(region[0], region[1], region[2])
X, Y = np.meshgrid(x, y)
zs = np.array(fun2(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
local_minima = detect_local_minima(Z)

# Plot objective function surface
surf1 = ax.plot_surface(X, Y, Z, alpha=0.5, label='a1')
# Plot minima
surf2 = ax.scatter(X[local_minima], Y[local_minima],
                   Z[local_minima], marker='.', color='r')

ax.set_xlabel('$g_1$')
ax.set_ylabel('$g_2$')
ax.set_zlabel('$L(g_1,g_2,\\tau)$')

ax.view_init(20, 30)
# ax.legend('bla')
# plt.show()

# Save to tikz
# Function already implemented in file
# surf1._facecolors2d = surf1._facecolor3d
# surf1._edgecolors2d = surf1._edgecolor3d
# surf2._facecolors2d = surf2._facecolor3d
# surf2._edgecolors2d = surf2._edgecolor3d
# plt.savefig("plots/3dplot_concreteMAP_1.pgf")
# tplt.save("plots/3dplot_concreteMAP_1.tikz")


plt.figure(2)
plt.grid()
for ind in range(0, local_minima[0].shape[0]):
    plt.plot(X[local_minima[0][ind], :], Z[local_minima[0][ind], :])
plt.xlabel('$g_1$')
plt.ylabel('$L(g_1,g_2,\\tau)$')

plt.figure(3)
plt.grid()
for ind in range(0, local_minima[0].shape[0]):
    plt.plot(Y[:, local_minima[1][ind]], Z[:, local_minima[1][ind]])
plt.xlabel('$g_2$')
plt.ylabel('$L(g_1,g_2,\\tau)$')


# Binary plot
def funbin(x):
    tau = 0.1  # 0.1
    alpha = 0.2  # 0.5
    noise_var = 0.25  # 2 ** 2
    yr = -1  # 0.4
    return (yr - np.tanh((x + np.log(1 / alpha - 1)) / (2 * tau))) ** 2 + 2 * noise_var * (x + 2 * np.log(1 + np.exp(-x)))


x0 = np.arange(-4, 4, region[2])
fun0 = funbin(x0)
peaks = find_peaks(-fun0)
plt.figure(4)
plt.grid()
plt.plot(x0, fun0, 'b-')
plt.scatter(x0[peaks[0]], fun0[peaks[0]], marker='.', color='r')
plt.xlabel('$s$')
plt.ylabel('$L(s,\\tau)$')


# Save to tikz
tplt.save("plots/3dplot_concreteMAP_2.tikz")
