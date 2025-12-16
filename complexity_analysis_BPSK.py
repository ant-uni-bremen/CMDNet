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

# Complexity analysis for BPSK MIMO transmission

# Settings
fn_tikz = ''    # 'bla'
Nt = 64
Nr = 64
L = 16
M = 4           # number of classes
# DetNet layers
Nv = 4 * Nt  # 2 (M = 2), 4 (M = 4), 4 (soft)
Nz = 12 * Nt  # 4 (M = 2), 8 (M = 4), 12 (soft)
Nrb = 1  # channel coherence interval

# Calculation of SD paper SNR
# EbN0 = 12
# var_sigma_dB = -EbN0 - 10 * np.log10(2 * np.log2(M))
# rho = 10 * np.log10((M ** 2 - 1)) - 10 * np.log10(12) - var_sigma_dB

if M == 2:
    comparison = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** (Nt) * (Nr * Nt + Nr + 2), 'k', True],
        # 'SD 8dB': [Nrb * M ** (0.3 * Nt) * (Nr * Nt + Nr + 2), 'k', True],
        # 'SD LB': [Nrb * M ** (Nt / 2 / ((M ** 2 - 1) / (6 * 1 / 10 ** (10 / 10)) + 1) ) / (M - 1), 'k', True],
        'SD 6dB': [Nrb * M ** (0.5 * Nt) * Nt, 'k', True],
        # 'SD 8dB': [Nrb * M ** (0.45 * Nt) * Nt, 'k', True],
        'SD 10dB': [Nrb * M ** (0.38 * Nt) * Nt, 'k', True],
        'SD 12dB': [Nrb * M ** (0.31 * Nt) * Nt, 'k', True],
        # 'SD 14dB': [Nrb * M ** (0.23 * Nt) * Nt, 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        # 'MMSE Inv': [np.min([Nrb * (Nt * Nr) + Nt ** 3 + 2 * Nt ** 2 * Nr + Nt * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt, Nrb * (Nt * Nr + Nt) + Nt ** 3 + 2 * Nt ** 2 * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt]), 'k', True],
        'MMSE': [np.min([Nrb * (Nt ** 2 * Nr + Nt * Nr + (2 * Nt ** 3 + 3 * Nt ** 2 - 5 * Nt) / 6), Nrb * (Nt ** 2 * Nr + Nt * Nr + Nt ** 2 + Nt) + 1 / 3 * Nt ** 3 + 2 / 3 * Nt ** 2 - Nt]), 'k', True],
        'MMSE OSIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'b', True],
        'AMP': [Nrb * L * (2 * Nt * Nr + 2 * Nt + Nr + 1), 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD': [Nrb * np.min([L * (Nt ** 2 + 3 * Nt + 4) + (Nt ** 2 + Nt) * Nr + 1, L * (2 * Nr * Nt + 3 * Nt + 4) + Nt * Nr + 1]), 'r', True],
        'DetNet': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [0.5 * 6 * Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }
else:
    comparison = {  # 'Tag': [# of multiplications, 'color in plot', on/off],
        'MAP': [Nrb * M ** Nt * (Nr * Nt + Nr + 2), 'k', True],
        # 'SD 6dB': [Nrb * M ** (0.3 * Nt) * Nt, 'k', True],
        'SD 8dB': [Nrb * M ** (0.25 * Nt) * Nt, 'k', True],
        # 'SD 10dB': [Nrb * M ** (0.21 * Nt) * Nt, 'k', True],
        # 'SD 12dB': [Nrb * M ** (0.19 * Nt) * Nt, 'k', True],
        'SD 14dB': [Nrb * M ** (0.13 * Nt) * Nt, 'k', True],
        'MF': [Nrb * Nt * Nr, 'k', True],
        'MMSE': [np.min([Nrb * (Nt * Nr) + Nt ** 3 + 2 * Nt ** 2 * Nr + Nt * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt, Nrb * (Nt * Nr + Nt) + Nt ** 3 + 2 * Nt ** 2 * Nr + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt]), 'k', True],
        'MMSE OSIC': [Nrb * (4/3 * Nt ** 3 + (4 * Nr - 5 / 4) * Nt ** 2 + (23 / 12 - 3 / 2 * Nr) * Nt) * 0.75 + (2 * Nt ** 2 + 4 * Nt * Nr - 2 * Nt) * 0.75, 'b', True],
        'AMP': [L * (2 * Nr * Nt + 6 * M * Nt + Nr) + Nr * Nt + 3, 'b', True],
        # 'MMNet': [Nrb * (L * (Nt * (Nt + 1 + 2 * M + 5 * Nr) + Nr + 2) + Nt * Nr + 1), 'o', True],
        'CMD': [Nrb * np.min([L * (Nt ** 2 + Nt * (4 * M + 1) + 2) + (Nt ** 2 + Nt) * Nr, L * (Nt * (4 * M + 2 * Nr + 1) + 2) + Nt * Nr]), 'r', True],
        'DetNet': [Nrb * np.min([L * (Nt * (Nt + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + (Nt ** 2 + Nt) * Nr, L * (Nt * (2 * Nr + 4 + Nz + M * (Nz + 1)) + 2 * Nv * Nz) + Nt * Nr]), 'm', True],
        'OAMPNet': [Nrb * (L * (2 * Nt * Nr + 2 * Nt + Nr + 1) + L * (Nt ** 3 + 2 * Nt ** 2 * Nr + Nt + 4 / 3 * Nt ** 3 + 5 / 3 * Nt ** 2 - Nt)), 'y', True],
        'SDR': [Nrb * np.max([Nr, Nt]) ** 4.5 * np.log(1 / 0.1), 'g', True],
        'SDR BQP': [0.5 * 6 * Nrb * np.max([Nr, Nt]) ** 3.5 * np.log(1 / 0.1), 'g', True],
    }


comp = comparison


# Performance curves

plt.figure(1)
for ind, key in enumerate(comp):
    # if key != 'MAP':
    plt.bar(key, comp[key][0], color=comp[key][1])


# plt.xlim(-6 + snr_shift, 40 + snr_shift)
plt.yscale('log')
plt.ylim(10 ** 2, 10 ** 12)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
plt.ylabel('# of Multiplication Ops')
# plt.legend()

# #plt.show() # comment if you want to save with tplt
tplt.save("plots/complexity_{}x{}_{}".format(Nt, Nr, L) + fn_tikz + ".tikz")
