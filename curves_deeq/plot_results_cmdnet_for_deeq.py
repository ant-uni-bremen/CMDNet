#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:35:17 2019

@author: beck
"""

# Include parent folder
import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA
sys.path.append('../..')                      # NOQA

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp


def dbinv(in_x):
    '''Converts SNR in dB back to normal scale
    '''
    return 10 ** (in_x / 10)


# Settings
y_axis = 'ber'      # ber, fer, ser, ce, mse # Only works with own simulations!
x_axis = 'ebn0'     # ebn0, snr, cebn0


# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

plt.figure()
# plt.title("BER vs Eb/N0")
# plt.xlabel("Eb/N0 (dB)")
# plt.ylabel("Bit Error Rate (BER)")
# plt.grid(True)

# Iterate through all .npz files
for filename in os.listdir(script_dir):
    if filename.endswith(".npz"):
        filepath = os.path.join(script_dir, filename)
        try:
            data = np.load(filepath)
            ebn0 = data[x_axis]
            ber = data[y_axis]

            plt.semilogy(ebn0, ber, marker='o',
                         label=os.path.splitext(filename)[0])
        except KeyError as e:
            print(f"Skipping {filename}: missing key {e}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

# Calculate AWGN curve that approximates BER curves for antenna dimensions growing to infinity
snr_awgn = np.arange(-6, 30, 1)
ber_awgn = 0.5 * sp.erfc(np.sqrt(dbinv(snr_awgn)))
plt.semilogy(snr_awgn, ber_awgn, 'g-', label='AWGN')


# Plot options
plt.xlim(-6, 30)
plt.ylim(10 ** -6, 1)
plt.grid(visible=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(visible=True, which='minor',
         color='#999999', linestyle='-', alpha=0.3)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.tight_layout()
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
