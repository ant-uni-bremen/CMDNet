#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  14 13:30:43 2020

@author: beck
"""

import numpy as np
# import scipy.special as sp
# import scipy.io as sio
import matplotlib.pyplot as plt
import tikzplotlib as tplt
# import os
# import myfunctions as mf

# Complexity analysis for QAM16 MIMO transmission

# Settings
fn_tikz = ''    # 'bla'
Nt = 64
Nr = 64
L = 16
M = 2           # number of classes
# DetNet layers
Nv = 2 * Nt * int(np.log2(M))  # 2, 4
Nz = 4 * Nt * int(np.log2(M))  # 4, 12
Nrb = 1  # channel coherence interval

if M == 2:
    comparison = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** (Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'SD 8dB': [Nrb * M ** (0.3 * Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        'MMSE bin': [np.min([Nrb * (Nt ** 2 * Nr + Nt * Nr + (2 * Nt ** 3 + 3 * Nt ** 2 - 5 * Nt) / 6), Nrb * (Nt ** 2 * Nr + Nt * Nr + Nt ** 2 + Nt) + 1 / 3 * Nt ** 3 + 2 / 3 * Nt ** 2 - Nt]), 'k', True],
        # 'MMSE SIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'p', True],
        'AMP bin': [Nrb * L * (2 * Nt * Nr + 2 * Nt + Nr + 1), 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD bin': [Nrb * np.min([L * (Nt ** 2 + 3 * Nt + 4) + (Nt ** 2 + Nt) * Nr + 1, L * (2 * Nr * Nt + 3 * Nt + 4) + Nt * Nr + 1]), 'r', True],
        'DetNet soft': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }
else:
    comparison = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** Nt * (Nr * Nt + Nr + 2), 'k', True],
        'SD 8dB': [Nrb * M ** (0.5 * Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        'MMSE Inv': [np.min([Nrb * (Nt * Nr) + Nt ** 3 + 2 * Nt ** 2 * Nr + Nt * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt, Nrb * (Nt * Nr + Nt) + Nt ** 3 + 2 * Nt ** 2 * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt]), 'k', True],
        # 'MMSE SIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'p', True],
        'AMP': [L * (2 * Nr * Nt + 6 * M * Nt + Nr) + Nr * Nt + 3, 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD': [Nrb * np.min([L * (Nt ** 2 + Nt * (4 * M + 1) + 2) + (Nt ** 2 + Nt) * Nr, L * (Nt * (4 * M + 2 * Nr + 1) + 2) + Nt * Nr]), 'r', True],
        'DetNet soft': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }

Nt = 16
Nr = 16
L = 16
M = 2           # number of classes
# DetNet layers
Nv = 2 * Nt * int(np.log2(M))  # 2, 4
Nz = 4 * Nt * int(np.log2(M))  # 4, 12
Nrb = 1  # channel coherence interval

if M == 2:
    comparison2 = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** (Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'SD 8dB': [Nrb * M ** (0.3 * Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        'MMSE bin': [np.min([Nrb * (Nt ** 2 * Nr + Nt * Nr + (2 * Nt ** 3 + 3 * Nt ** 2 - 5 * Nt) / 6), Nrb * (Nt ** 2 * Nr + Nt * Nr + Nt ** 2 + Nt) + 1 / 3 * Nt ** 3 + 2 / 3 * Nt ** 2 - Nt]), 'k', True],
        # 'MMSE SIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'p', True],
        'AMP bin': [Nrb * L * (2 * Nt * Nr + 2 * Nt + Nr + 1), 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD bin': [Nrb * np.min([L * (Nt ** 2 + 3 * Nt + 4) + (Nt ** 2 + Nt) * Nr + 1, L * (2 * Nr * Nt + 3 * Nt + 4) + Nt * Nr + 1]), 'r', True],
        'DetNet soft': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }
else:
    comparison2 = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** Nt * (Nr * Nt + Nr + 2), 'k', True],
        'SD 8dB': [Nrb * M ** (0.5 * Nt) * (Nr * Nt + Nr + 2), 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        'MMSE Inv': [np.min([Nrb * (Nt * Nr) + Nt ** 3 + 2 * Nt ** 2 * Nr + Nt * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt, Nrb * (Nt * Nr + Nt) + Nt ** 3 + 2 * Nt ** 2 * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt]), 'k', True],
        # 'MMSE SIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'p', True],
        'AMP': [L * (2 * Nr * Nt + 6 * M * Nt + Nr) + Nr * Nt + 3, 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD': [Nrb * np.min([L * (Nt ** 2 + Nt * (4 * M + 1) + 2) + (Nt ** 2 + Nt) * Nr, L * (Nt * (4 * M + 2 * Nr + 1) + 2) + Nt * Nr]), 'r', True],
        'DetNet soft': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }

comp = comparison

comp2 = comparison2

# Performance curves

labels = []
bar1 = []
bar2 = []

plt.figure(1)
for ind, key in enumerate(comp):
    # if key != 'MAP':
    # plt.bar(key, comp[key][0], color = comp[key][1])
    labels.append(key)
    bar1.append(comp[key][0])

for ind, key in enumerate(comp2):
    # if key != 'MAP':
    # plt.bar(key, comp[key][0], color = comp[key][1])
    bar2.append(comp2[key][0])


x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, bar1, width, label='Nt = 64')
rects2 = ax.bar(x + width/2, bar2, width, label='Nt = 16')

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_title('Scores by group and gender')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()


# plt.xlim(-6 + snr_shift, 40 + snr_shift)
plt.yscale('log')
plt.ylim(10 ** 2, 10 ** 10)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('# of Multiplication Ops')
# plt.legend()

# #plt.show() # comment if you want to save with tplt
# tplt.save("plots/MIMO_"+mod+"_{}x{}_{}".format(Nt, Nr, L) + fn_tikz + ".tikz")
