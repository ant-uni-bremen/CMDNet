#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:35:17 2019

@author: beck
"""

import numpy as np
import scipy.special as sp
import scipy.io as sio
import matplotlib.pyplot as plt
import tikzplotlib as tplt
import os
## Own packages
# Include parent folder
import sys
sys.path.append('..')
import myfunctions as mf

## Merge results of SD evaluated for different single SNRs

# Settings
Nt = 64
Nr = 128
L = 64
mod = 'QPSK'
rho = 10
cell_sector = '_120'

y_axis = 'ber' # ber, fer, ser, ce, mse # Only works with own simulations!
x_axis = 'ebn0' # ebn0, snr
path = os.path.join('curves', mod,'{}x{}'.format(Nt, Nr))
save = mf.savemodule('npz')

if rho == 0:
    rho_ext = ''
elif rho > 1:
    rho_ext = '_OneRing{}'.format(rho) + cell_sector
else:
    rho_ext = '_rho0{}'.format(rho)

# Merge SD data
ber_tot = []
fer_tot = []
ser_tot = []
ebn0_tot = []
snr_tot = []
for snr_i in range(-6, 20, 1): # range(-6, 20)
    one_file = 0
    ebn0 = []
    snr = []
    ber = []
    Nerr = []
    Nit = []
    #bitanz = []
    Nb = []
    fn_ext = rho_ext + '_{}dB'.format(snr_i)
    for ii in range(0, 20):
        fn = os.path.join(path, 'BER_SD_' + mod + '_{}_{}'.format(Nt, Nr) + fn_ext + str(ii) + '.mat')
        if os.path.isfile(fn):
            print(fn)
            one_file = 1
            load_data = sio.loadmat(fn)['BER'][0, 0][['EbN0', 'SNR', 'SD', 'Nit', 'Nerr', 'Nb']] #, 'BitAnz']]
            ebn0.append(load_data[0][0,:])
            snr.append(load_data[1][0,:])
            ber.append(load_data[2])
            Nit.append(load_data[3][0, :][0])
            Nerr.append(load_data[4][0, :][0])
            #bitanz.append(load_data[6][0])
            Nb.append(load_data[5][0])
    if one_file:
        bitanz = np.array(Nb)[:,0].astype('uint64') * Nt
        Nbits = bitanz * np.array(Nit).astype('uint64')
        # ber_tot2 = np.sum(Nerr) / np.sum(Nbits)
        # result for one snrs
        ber_tot.append(np.sum(np.array(ber)[:, 0, 0] * Nbits) / np.sum(Nbits))
        fer_tot.append(np.sum(np.array(ber)[:, 1, 0] * Nbits) / np.sum(Nbits))
        ser_tot.append(np.sum(np.array(ber)[:, 2, 0] * Nbits) / np.sum(Nbits))
        ebn0_tot.append(ebn0[0][0])
        snr_tot.append(snr[0][0])
        print('EbN0: {}dB, Nerr: {}, Nit: {}, BER: {:.2e}'.format(ebn0[0][0], np.sum(Nerr), np.sum(np.array(Nit).astype('uint64')), ber_tot[-1]))




# result for all snrs
zero_list = [0 for ii in range(0, len(ber_tot))]
results = {
    "ebn0": ebn0_tot,
    "cebn0": ebn0_tot,
    "snr": snr_tot - 10 * np.log10(2), # correction term
    "ber": ber_tot,
    "cber": zero_list,
    "ser": ser_tot,
    "fer": fer_tot,
    "cfer": zero_list,
    "ce": zero_list,
    "mse": zero_list,
    }
pathfile = os.path.join(path, 'RES_SD_' + mod + '_{}_{}'.format(Nt, Nr) + rho_ext)
save.save(pathfile, results, verbose = 1)



#EOF