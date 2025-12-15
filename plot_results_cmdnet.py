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

import numpy as np
import scipy.special as sp
import scipy.io as sio
import matplotlib.pyplot as plt
import tikzplotlib as tplt
import os

# Own packages
import utilities.my_functions as mf
import utilities.my_communications as com
import utilities.my_math_operations as mop

# Settings
mode = 0            # 0: default, 1:codes
mode_online = 0     # 0: default, 1: online training
fn_tikz = ''        # '_codeLDPC', '_corr', '_codeLDPC_fer'
Nt = 8
Nr = 8
L = 8
mod = 'QPSK'
y_axis = 'ber'      # ber, fer, ser, ce, mse # Only works with own simulations!
x_axis = 'ebn0'     # ebn0, snr, cebn0
path = os.path.join('curves', mod, '{}x{}'.format(Nt, Nr))
load = mf.savemodule()


# Plot tables

bin_comparison = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'AMP BPSK': ['RES_AMP_'+'BPSK'+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b->', y_axis, True],
    'DetNet softmax': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr_softmax'.format(Nt, Nr, L), 'm-o', y_axis, True],
    # 'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'm--<', y_axis, True],
    'DetNet HD': ['RES_DetNet_'+mod+'_{}_{}_{}_Nv2KNz4K_HD'.format(Nt, Nr, L), 'm--<', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'CMD bin_': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin deeq': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_deeq'.format(Nt, Nr, L), 'r-<', y_axis, True],
    'CMD bin mismatch': ['RES_CMD_'+mod+'_{}_{}_64_binary_mismatch64x64_binary_tau0.1'.format(Nt, Nr), 'r--<', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r--o', y_axis, True],
    'CMDpar': ['RES_CMDpar_'+mod+'_{}_{}_16_binary_test2'.format(Nt, Nr), 'r-s', y_axis, True],
    'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r-', y_axis, True],
    # 'CMD': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'r-x', y_axis, True],
    'HyperCMD': ['RES_HyperCMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k-o', y_axis, True],
    # 'CMD tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.01'.format(Nt, Nr, L), 'r->', y_axis, True],
    'CMD bin snr1_24': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr1_24'.format(Nt, Nr, L), 'r-->', y_axis, True],
    'CMD bin snr7_30': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr7_30'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD bin snr4_11': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr4_11'.format(Nt, Nr, L), 'r--x', y_axis, True],
    'CMD bin snr4_11 splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr4_11_splin'.format(Nt, Nr, L), 'r--o', y_axis, True],
    'OAMPNet': ['RES_OAMPNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'OAMPNet snr4_11': ['RES_OAMPNet_'+mod+'_{}_{}_{}_snr4_11'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'SDR': ['RES_SDR_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'g-o', y_axis, True],
}
cmd_comparison = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin_': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'CMD tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.01'.format(Nt, Nr, L), 'r->', y_axis, True],
    'CMD bin snr1_24': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr1_24'.format(Nt, Nr, L), 'r-->', y_axis, True],
    'CMD bin snr7_30': ['RES_CMD_'+mod+'_{}_{}_{}_binary_snr7_30'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r-', y_axis, True],
    # 'CMD taumax2': ['RES_CMD_'+mod+'_{}_{}_{}_binary_taumax2'.format(Nt, Nr, L), 'r--', y_axis, True],
    # 'CMD bin lin tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'g-<', y_axis, True],
    'CMD bin lin tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'CMD bin splin fixed tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splinfix_tau0.01'.format(Nt, Nr, L), 'b-', y_axis, True],
    'CMD bin spdef fixed tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_spdeffix_tau0.1'.format(Nt, Nr, L), 'g-x', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'OAMPNet': ['RES_OAMPNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'y-o', y_axis, True],
    # 'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'c-x', y_axis, True],
    # 'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-<', y_axis, True],
}
comparison = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    # 'CMD': ['RES_CMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD snr10_33': ['RES_CMD_'+mod+'_{}_{}_{}_snr10_33'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD mismatch bin': ['RES_CMD_'+mod+'_{}_{}_{}_mismatch_binary_tau0.1'.format(Nt, Nr, L), 'r--<', y_axis, True],
    'CMD convex snr10_33': ['RES_CMD_'+mod+'_{}_{}_{}_convex'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'CMD convex softmax': ['RES_CMD_'+mod+'_{}_{}_{}_losssoftmax'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD taumax1 snr7_30': ['RES_CMD_'+mod+'_{}_{}_{}_taumax1_snr7_30'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD tauscale': ['RES_CMD_'+mod+'_{}_{}_{}_tauscale'.format(Nt, Nr, L), 'r-<', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'DetNet Hscal': ['RES_DetNet_'+mod+'_{}_{}_{}_Hscal2'.format(Nt, Nr, L), 'm-x', y_axis, True],
    'DetNet snr7_30': ['RES_DetNet_'+mod+'_{}_{}_{}_snr_7_30'.format(Nt, Nr, L), 'm-->', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'MMNet ON': ['RES_MMNet_'+mod+'_{}_{}_{}_snr9_30_Nb1500'.format(Nt, Nr, L), 'c--o', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'OAMPNet': ['RES_OAMPNet_'+mod+'_{}_{}_{}_snr7_30_Nb1500'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'SDR': ['RES_SDR_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'g-o', y_axis, True],
}
layer = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin 1 tau0.1': ['RES_CMD_'+mod+'_{}_{}_1_binary_tau0.1'.format(Nt, Nr), 'g--', y_axis, True],
    'CMD bin 2 tau0.1': ['RES_CMD_'+mod+'_{}_{}_2_binary_tau0.1'.format(Nt, Nr), 'r--', y_axis, True],
    'CMD bin 4 tau0.1': ['RES_CMD_'+mod+'_{}_{}_4_binary_tau0.1'.format(Nt, Nr), 'k--', y_axis, True],
    'CMD bin 8 tau0.1': ['RES_CMD_'+mod+'_{}_{}_8_binary_tau0.1'.format(Nt, Nr), 'g-', y_axis, True],
    'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'b-', y_axis, True],
    'CMD bin 32 tau0.1': ['RES_CMD_'+mod+'_{}_{}_32_binary_tau0.1'.format(Nt, Nr), 'y-', y_axis, True],
    'CMD bin 64 tau0.1': ['RES_CMD_'+mod+'_{}_{}_64_binary_tau0.1'.format(Nt, Nr), 'r-', y_axis, True],
    'CMD bin 128 tau0.1': ['RES_CMD_'+mod+'_{}_{}_128_binary_tau0.1'.format(Nt, Nr), 'k-', y_axis, True],
}
layer2 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'AMP bin64': ['RES_AMP_'+mod+'_{}_{}_64_binary'.format(Nt, Nr), 'b-<', y_axis, True],
    'AMP 64': ['RES_AMP_'+mod+'_{}_{}_64'.format(Nt, Nr), 'b-x', y_axis, True],
    'AMP 32': ['RES_AMP_'+mod+'_{}_{}_32'.format(Nt, Nr), 'b-x', y_axis, True],
    'AMP 16': ['RES_AMP_'+mod+'_{}_{}_16'.format(Nt, Nr), 'b-o', y_axis, True],
    'AMP 8': ['RES_AMP_'+mod+'_{}_{}_8'.format(Nt, Nr), 'b--', y_axis, True],
    'AMP 4': ['RES_AMP_'+mod+'_{}_{}_4'.format(Nt, Nr), 'b-*', True],
}
unfoldvsplain = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin 64 tau0.1': ['RES_CMD_'+mod+'_{}_{}_64_binary_tau0.1'.format(Nt, Nr), 'r-o', y_axis, True],
    'CMD bin spdef fixed tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_spdeffix_tau0.1'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'CMD bin splin fixed tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splinfix_tau0.1'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'CMD bin spcon fixed': ['RES_CMD_'+mod+'_{}_{}_{}_binary_spconfix'.format(Nt, Nr, L), 'k-o', y_axis, True],
    'CMD bin splin fixed tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splinfix_tau0.01'.format(Nt, Nr, L), 'b-', y_axis, True],
    'CMD bin spdef fixed tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_spdeffix_tau0.01'.format(Nt, Nr, L), 'g-', y_axis, True],
}
type_sp = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin def tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-', y_axis, True],
    'CMD bin def tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin lin tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'b-', y_axis, True],
    'CMD bin lin tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'CMD bin const': ['RES_CMD_'+mod+'_{}_{}_{}_binary_spconst'.format(Nt, Nr, L), 'k-o', y_axis, True],
}
value_sp = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    # 'CMD bin 64 tau0.01': ['RES_CMD_'+mod+'_{}_{}_64_binary'.format(Nt, Nr), 'r-', y_axis, True],
    # 'CMD bin 64 tau0.1': ['RES_CMD_'+mod+'_{}_{}_64_binary_tau0.1'.format(Nt, Nr), 'k-', y_axis, True],
    # 'CMD bin 64 tau0.075': ['RES_CMD_'+mod+'_{}_{}_64_binary_tau0.075'.format(Nt, Nr), 'b-x', y_axis, True],
    'CMD bin 16': ['RES_CMD_'+mod+'_{}_{}_16_binary'.format(Nt, Nr), 'b-', y_axis, True],
    'CMD bin 16 tau0.01': ['RES_CMD_'+mod+'_{}_{}_16_binary_correct'.format(Nt, Nr), 'r-', y_axis, True],
    # 'CMD bin 16 tau0.05': ['RES_CMD_'+mod+'_{}_{}_16_binary_correct2'.format(Nt, Nr), 'g-x', y_axis, True],
    'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'y--', y_axis, True],
    'CMD bin 16 tau0.1': ['RES_CMD_'+mod+'_{}_{}_16_binary_correct3'.format(Nt, Nr), 'g-', y_axis, True],
    # 'CMD bin 16 tau0.125': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.125'.format(Nt, Nr), 'y-x', y_axis, True],
    # 'CMD bin 16 tau0.15': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.15'.format(Nt, Nr), 'y-o', y_axis, True],
    # 'CMD bin 16 tau0.2': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.2'.format(Nt, Nr), 'y-<', y_axis, True],
    # 'CMD bin 8 tau0.075': ['RES_CMD_'+mod+'_{}_{}_8_binary_tau0.075'.format(Nt, Nr), 'b-', y_axis, True],
    # 'CMD bin 8 tau0.1': ['RES_CMD_'+mod+'_{}_{}_8_binary_tau0.1'.format(Nt, Nr), 'y-', y_axis, True],
}
code_ldpc = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-x', 'ebn0', 'fer', True],
    # 'CMD bin splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'r-x', 'ebn0', 'ber', True],
    'CMD bin LDPC bp horiz': ['RES_CMD_QPSK_64_64_64_LDPC64x128bphoriz_binary_tau0.1_float64', 'r-o', 'cebn0', 'cfer', True],
    'CMD bin LDPC bp horiz float64 2': ['RES_CMD_QPSK_64_64_64_LDPC64x128bphoriz_binary_tau0.1_float64_2', 'r-.o', 'cebn0', 'cfer', True],
    'CMD bin LDPC bp horiz float32': ['RES_CMD_QPSK_64_64_64_LDPC64x128bphoriz_binary_tau0.1', 'r-.s', 'cebn0', 'cfer', True],
    'CMD bin LDPC bp horiz splin': ['RES_CMD_QPSK_64_64_64_LDPC64x128bphoriz_binary_splin', 'r--<', 'cebn0', 'cfer', True],
    'CMD bin LDPC bp horiz splinfixed': ['RES_CMD_QPSK_64_64_64_LDPC64x128bphoriz_binary_splinfix_tau0.01', 'r--x', 'cebn0', 'cfer', True],
    'AMP bin LDPC bp horiz': ['RES_AMP_QPSK_64_64_64_LDPC64x128bphoriz', 'g-x', 'cebn0', 'cfer', True],
    'MMSE LDPC bp horiz': ['RES_MMSE_QPSK_64_64_64_LDPC64x128bphoriz', 'k->', 'cebn0', 'cfer', True],
    'DetNet LDPC bp horiz': ['RES_DetNet_'+mod+'_{}_{}_{}_LDPC64x128bphoriz_defsnr_softmax'.format(Nt, Nr, L), 'm-o', 'cebn0', 'cfer', True],
    'DetNet LDPC bp horiz orig': ['RES_DetNet_'+mod+'_{}_{}_{}_LDPC64x128bphoriz_defsnr'.format(Nt, Nr, L), 'm-<', 'cebn0', 'cfer', True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', 'ebn0', 'fer', True],
}
code_ham = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-x', 'ebn0', 'ber', True],
    # 'CMD bin splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'r-x', 'ebn0', 'ber', True],
    'CMD bin ham bp horiz': ['RES_CMD_QPSK_64_64_64_hamming4x7bphoriz_binary_tau0.1', 'r-o', 'cebn0', 'cber', True],
    'CMD bin ham bp horiz splin': ['RES_CMD_QPSK_64_64_64_hamming4x7bphoriz_binary_splin', 'r--<', 'cebn0', 'cber', True],
    'AMP bin ham bp horiz': ['RES_AMP_QPSK_64_64_64_hamming4x7bphoriz', 'g-o', 'cebn0', 'cber', True],
    'MMSE ham bp horiz': ['RES_MMSE_QPSK_64_64_64_hamming4x7bphoriz', 'k->', 'cebn0', 'cber', True],
    'DetNet ham bp horiz': ['RES_DetNet_'+mod+'_{}_{}_{}_hamming4x7bphoriz_defsnr'.format(Nt, Nr, L), 'm-o', 'cebn0', 'cber', True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', 'ebn0', 'ber', True],
}
train = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'CMD bin_': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin mse': ['RES_CMD_'+mod+'_{}_{}_{}_binary_mseloss'.format(Nt, Nr, L), 'r--<', y_axis, True],
    'CMD bin multi': ['RES_CMD_'+mod+'_{}_{}_{}_binary_multiloss'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD bin mse multi': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_mse_multiloss'.format(Nt, Nr, L), 'r:', y_axis, True],
    # 'CMD': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'CMD CE': ['RES_CMD_'+mod+'_{}_{}_{}_losssoftmax'.format(Nt, Nr, L), 'r-*', y_axis, True],
    # 'CMD CE taumax2': ['RES_CMD_'+mod+'_{}_{}_{}_taumax2_losssoftmax'.format(Nt, Nr, L), 'r--x', y_axis, True],
    # 'CMD taumax2': ['RES_CMD_'+mod+'_{}_{}_{}_taumax2'.format(Nt, Nr, L), 'r--s', y_axis, True],
    'CMD CE tau0.1 old': ['RES_CMD_'+mod+'_{}_{}_{}_softmax_tau0.1_2'.format(Nt, Nr, L), 'r-*', y_axis, True],
    # 'CMD bin Nb5000': ['RES_CMD_'+mod+'_{}_{}_{}_binary_Nb5000'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
}
detnet = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'DetNet 1res': ['BER_DetNet_'+mod+'_{}_{}_{}_snr4_27'.format(Nt, Nr, L), 'm--', 'ber', True],
    'DetNet 1res CMDopt': ['BER_DetNet_'+mod+'_{}_{}_{}_snr4_27_CMDopt'.format(Nt, Nr, L), 'g--', 'ber', True],
    'DetNet 1res snrdef': ['BER_DetNet_'+mod+'_{}_{}_{}_snr4_11_defsnr'.format(Nt, Nr, L), 'b--', 'ber', True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'DetNet CMDopt': ['RES_DetNet_'+mod+'_{}_{}_{}_CMDopt'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'DetNet snrdef': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'b-o', y_axis, True],
}
detnet16qam = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'DetNet defsnr': ['BER_DetNet_'+mod+'_{}_{}_{}_snr2_7_defsnr'.format(Nt, Nr, L), 'm--', 'ber', True],
    'DetNet 1res snr4_27': ['BER_DetNet_'+mod+'_{}_{}_{}_snr4_27'.format(Nt, Nr, L), 'g--', 'ber', True],
    'DetNet 1res nonoise': ['BER_DetNet_'+mod+'_{}_{}_{}_snr10_33_nonoise'.format(Nt, Nr, L), 'b--', 'ber', True],
    'DetNet 1res snr10_33': ['BER_DetNet_'+mod+'_{}_{}_{}_snr10_33_snr10_33'.format(Nt, Nr, L), 'r--', 'ber', True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'DetNet CMDopt': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'DetNet snrdef': ['RES_DetNet_'+mod+'_{}_{}_{}_nonoise'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'DetNet snr10_33': ['RES_DetNet_'+mod+'_{}_{}_{}_snr10_33'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'DetNet 64x128 0': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr0'.format(Nt, Nr, L), 'c-o', y_axis, True],
    'DetNet Hscal': ['RES_DetNet_'+mod+'_{}_{}_{}_Hscal2'.format(Nt, Nr, L), 'c-o', y_axis, True],
    'DetNet test0': ['RES_DetNet_'+mod+'_{}_{}_{}_test0'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'DetNet test0_s': ['RES_DetNet_'+mod+'_{}_{}_{}_test0_s'.format(Nt, Nr, L), 'c-<', y_axis, True],
    'DetNet ascal': ['RES_DetNet_'+mod+'_{}_{}_{}_ascal'.format(Nt, Nr, L), 'b-<', y_axis, True],
}
mmnet = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MMNet 1res def': ['SER_MMNet_'+mod+'_{}_{}_10_snr4_27_default'.format(Nt, Nr), 'c-o', 'ser', True],
    'MMNet def': ['RES_MMNet_'+mod+'_{}_{}_10_default'.format(Nt, Nr), 'm-o', y_axis, True],
    'MMNet 1res CMD': ['SER_MMNet_'+mod+'_{}_{}_{}_snr4_27'.format(Nt, Nr, L), 'c-x', 'ser', True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'm-x', y_axis, True],
    'MMNet 1res detnet': ['SER_MMNet_'+mod+'_{}_{}_{}_snr4_11_detnet'.format(Nt, Nr, L), 'c-<', 'ser', True],
    'MMNet detnet': ['RES_MMNet_'+mod+'_{}_{}_{}_detnet'.format(Nt, Nr, L), 'm-<', y_axis, True],
}
corr_rho05 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_rho05'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho05'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin snr10_33': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho05_snr10_33'.format(Nt, Nr, L), 'r--', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'm--o', y_axis, True],
    'DetNet snr10_33': ['RES_DetNet_'+mod+'_{}_{}_{}_rho05_snr10_17'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'MMNet snr10_33': ['RES_MMNet_'+mod+'_{}_{}_{}_rho05_snr10_33'.format(Nt, Nr, L), 'c--<', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary_rho06'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'k-<', y_axis, True],
    # 'MMSE rho=0.6': ['RES_MMSE_'+mod+'_{}_{}_{}_rho06'.format(Nt, Nr, L), 'k--x', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'b->', y_axis, True],
}
corr_rho07 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_rho07'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin snr8_31': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07_snr8_31'.format(Nt, Nr, L), 'r-', y_axis, True],
    # 'CMD bin snr4_11': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07_snr4_11'.format(Nt, Nr, L), 'r--o', y_axis, True],
    'CMD bin splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07_splin'.format(Nt, Nr, L), 'r--x', y_axis, True],
    # 'CMD bin tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07_tau0.01'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'CMD bin taumax2': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rho07_taumax2'.format(Nt, Nr, L), 'r->', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'm--o', y_axis, True],
    # 'DetNet rho=0.5': ['RES_DetNet_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'm--o', y_axis, True],
    'DetNet snr24_47': ['RES_DetNet_'+mod+'_{}_{}_{}_rho07_snr24_47'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'MMNet snr24_47': ['RES_MMNet_'+mod+'_{}_{}_{}_rho07_snr24_47'.format(Nt, Nr, L), 'c--<', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary_rho05'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'k--', y_axis, True],
    # 'MMSE rho=0': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-o', y_axis, True],
    'MMSE rho=0.7': ['RES_MMSE_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'k-<', y_axis, True],
    # 'MMSE rho=0.6': ['RES_MMSE_'+mod+'_{}_{}_{}_rho06'.format(Nt, Nr, L), 'k--x', y_axis, True],
    # 'MMSE rho=0.5': ['RES_MMSE_'+mod+'_{}_{}_{}_rho05'.format(Nt, Nr, L), 'k-->', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_rho07'.format(Nt, Nr, L), 'b->', y_axis, True],
}
bin_cmd_orig = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'm-o', y_axis, True],
    # 'CMD bin_': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'CMD tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_tau0.01'.format(Nt, Nr, L), 'r->', y_axis, True],
    'CMD bin gradL spdef': ['RES_CMD_'+mod+'_{}_{}_{}_binary_gradL'.format(Nt, Nr, L), 'r-->', y_axis, True],
    'CMD bin gradL spconst': ['RES_CMD_'+mod+'_{}_{}_{}_binary_gradLspconst'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD bin gradL splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_gradLsplin'.format(Nt, Nr, L), 'r--x', y_axis, True],
}
one_ring = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_OneRing20'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing20'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin snr7_36': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing20_snr7_36'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'DetNet softmax': ['RES_DetNet_'+mod+'_{}_{}_{}_OneRing20_softmax'.format(Nt, Nr, L), 'm--x', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'b->', y_axis, True],
    'OAMPNet snr4_11': ['RES_OAMPNet_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'SDR': ['RES_SDR_'+mod+'_{}_{}_{}_OneRing20'.format(Nt, Nr, L), 'g-o', y_axis, True],
}
one_ring2 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_OneRing20_120'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing20_120'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD bin snr7_36': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing20_120_snr7_36'.format(Nt, Nr, L), 'r-x', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_OneRing20_120_snr4_27'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'b->', y_axis, True],
    'SDR': ['RES_SDR_'+mod+'_{}_{}_{}_OneRing20_120'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'OAMPNet snr4_27': ['RES_OAMPNet_'+mod+'_{}_{}_{}_OneRing20_120_snr4_27'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'CMD bin mismatch iid': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing20_120_mismatchparams_iid_snr4_11'.format(Nt, Nr, L), 'r-*', y_axis, True],
}
one_ring3 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_OneRing10_120'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_OneRing10_120'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'm-o', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_OneRing10_120_snr4_27'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'b->', y_axis, True],
    'SDR': ['RES_SDR_'+mod+'_{}_{}_{}_OneRing10_120'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'OAMPNet snr4_27': ['RES_OAMPNet_'+mod+'_{}_{}_{}_OneRing10_120_snr4_27'.format(Nt, Nr, L), 'y-o', y_axis, True],
}
rxrho07 = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}_binary_rxrho07'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rxrho07'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin snr4_11 splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_rxrho07_snr4_11_splin'.format(Nt, Nr, L), 'r--x', y_axis, True],
    'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'm--o', y_axis, True],
    'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'c-x', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'k--', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'k-<', y_axis, True],
    'MFVI par': ['RES_MFVIpar_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'MFVI seq': ['RES_MFVIseq_'+mod+'_{}_{}_{}_rxrho07'.format(Nt, Nr, L), 'b->', y_axis, True],
}
cmdpar = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-<', y_axis, True],
    # 'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r--o', y_axis, True],
    'CMDpar 16': ['RES_CMDpar_'+mod+'_{}_{}_16_binary_test2'.format(Nt, Nr), 'r-s', y_axis, True],
    'CMDpar': ['RES_CMDpar_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-x', y_axis, True],
    # 'HyperCMD': ['RES_HyperCMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'k-o', y_axis, True],
    'OAMPNet': ['RES_OAMPNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'y-o', y_axis, True],
}
hypercmd = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    # 'CMD bin lin tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'g-o', y_axis, True],
    # 'CMD bin splin fixed tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splinfix_tau0.01'.format(Nt, Nr, L), 'b-', y_axis, True],
    # 'DetNet': ['RES_DetNet_'+mod+'_{}_{}_{}_defsnr'.format(Nt, Nr, L), 'm-o', y_axis, True],
    # 'OAMPNet': ['RES_OAMPNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'y-o', y_axis, True],
    'HyperCMD': ['RES_HyperCMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'CMD bin 16 tau0.075': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r-', y_axis, True],
    'CMD bin 16x16': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r--', y_axis, True],
    # 'HyperCMD 16x16 NL=8': ['RES_HyperCMD_'+mod+'_{}_{}_8_test'.format(Nt, Nr), 'b--', y_axis, True],
    # 'HyperCMD 16x16 new NL=8': ['RES_HyperCMD_'+mod+'_{}_{}_8_binary'.format(Nt, Nr), 'b-o', y_axis, True],
    'CMD bin 16': ['RES_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r--', y_axis, True],
    'HyperCMD 16': ['RES_HyperCMD_'+mod+'_{}_{}_16_test'.format(Nt, Nr), 'b--', y_axis, True],
    # 'HyperCMD sgd': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_sgd'.format(Nt, Nr, L), 'b-', y_axis, True],
    # 'HyperCMD bin Ne=3*10^5': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_Ne300000'.format(Nt, Nr, L), 'b-x', y_axis, True],
    # 'HyperCMD bin mult nonabs lr10-4': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'HyperCMD QR mult nonabs': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_QR_lr10-4_mult'.format(Nt, Nr, L), 'g--', y_axis, True],
    # 'HyperCMD QR mult2 abs': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_QR_lr10-4_mult2'.format(Nt, Nr, L), 'g--x', y_axis, True],
    # 'HyperCMD QR nonabs': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_QR_lr10-4'.format(Nt, Nr, L), 'g-o', y_axis, True],
    # 'HyperCMD QR lr10-3': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_QR_lr10-3'.format(Nt, Nr, L), 'g-*', y_axis, True],
    # 'HyperCMD QR abs': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_QR'.format(Nt, Nr, L), 'g-x', y_axis, True],
    'HyperCMD abs': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_abs'.format(Nt, Nr, L), 'b-x', y_axis, True],
    # 'HyperCMD abs Ne=3*10^5': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary2_abs'.format(Nt, Nr, L), 'k--', y_axis, True],
    # 'HyperCMD abs var': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary2_abs_var'.format(Nt, Nr, L), 'b-*', y_axis, True],
    'HyperCMD parvec': ['RES_HyperCMD_'+mod+'_{}_{}_{}_binary_parvec'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'HyperCMD parvec 8 lr10-3': ['RES_HyperCMD_'+mod+'_{}_{}_8_binary_parvec_lr10-3'.format(Nt, Nr, L), 'b-*', y_axis, True],
    # 'HyperCMD parvec 8': ['RES_HyperCMD_'+mod+'_{}_{}_8_binary_parvec'.format(Nt, Nr), 'b-<', y_axis, True],
    # 'HyperCMD parvec 4': ['RES_HyperCMD_'+mod+'_{}_{}_4_binary_parvec'.format(Nt, Nr), 'b--', y_axis, True],
    # 'MMNet CMD': ['RES_MMNet_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'c-x', y_axis, True],
    # 'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    # 'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-<', y_axis, True],
}
bpsk_comparison = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'AMP': ['RES_AMP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-o', y_axis, True],
    'AMP BPSK': ['RES_AMP_'+'BPSK'+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'CMD bin QPSK': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin BPSK': ['RES_CMD_'+'BPSK'+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k--o', y_axis, True],
    'LS BPSK': ['RES_LS_'+'BPSK'+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k--x', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-o', y_axis, True],
    'MMSE BPSK': ['RES_MMSE_'+'BPSK'+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-x', y_axis, True],
}
sgd = {  # 'Tag': ['data name', 'color in plot', y_axis, on/off],
    'MAP': ['RES_MAP_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-', y_axis, True],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, True],
    'CMD bin Nb=32': ['RES_CMD_'+mod+'_{}_{}_{}_binary_Nb32'.format(Nt, Nr, L), 'r-x', y_axis, True],
    'CMD bin Nb=5000': ['RES_CMD_'+mod+'_{}_{}_{}_binary_Nb5000'.format(Nt, Nr, L), 'r--', y_axis, True],
    'CMD bin lin tau0.01': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'g-o', y_axis, True],
    'CMD bin lin tau0.1': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'g--x', y_axis, True],
    'CMD sgd Nb=500 splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_sgd_splin'.format(Nt, Nr, L), 'k-', y_axis, True],
    # 'CMD sgd Nb=500 splin 2': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_sgdNb500_splin_tau0.1_wo_nesterov'.format(Nt, Nr, L), 'k--s', y_axis, True],
    # 'CMD sgd load': ['RES_CMD_'+mod+'_{}_{}_{}_binary_sgdload'.format(Nt, Nr, L), 'g-', y_axis, True],
    # 'CMD sgd spconst': ['RES_CMD_'+mod+'_{}_{}_{}_binary_sgd_spconst'.format(Nt, Nr, L), 'b-', y_axis, True],
    'CMD sgd Nb=32 splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_sgdNb32_splin'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'CMD sgd Nb=5000 splin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_sgdNb5000_splin'.format(Nt, Nr, L), 'k--', y_axis, True],
    'CMD sgdmom Nb=500 splin nesterov': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_sgdmomNb500'.format(Nt, Nr, L), 'b-', y_axis, True],
    # 'CMD sgdmom Nb=500 splin wo nesterov': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_sgdmomNb500_splin_tau0.1_wo_nesterov'.format(Nt, Nr, L), 'b--s', y_axis, True],
    'CMD sgdmom Nb=32 splin nesterov': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_sgdmomNb32'.format(Nt, Nr, L), 'b-x', y_axis, True],
    'CMD sgdmom Nb=5000 splin nesterov': ['RES_CMD_'+mod+'_{}_{}_{}_binary_splin_sgdmomNb5000'.format(Nt, Nr, L), 'b--', y_axis, True],
}
online = {  # 'Tag': ['data name', 'color in plot', y_axis, online on/off, on/off],
    'SD': ['RES_SD_'+mod+'_{}_{}'.format(Nt, Nr), 'k-o', y_axis, False, True],
    'MF': ['RES_MF_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-x', y_axis, True],
    'LS': ['RES_LS_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k--o', y_axis, True],
    'MMSE': ['RES_MMSE_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'k-o', y_axis, True],
    'CMD bin': ['RES_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-o', y_axis, False, True],
    # 'CMD bin 2': ['RES_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r--o', y_axis, True],
    'CMD bin online Ne=100 snr1 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr1_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr2 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr2_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr3 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr3_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr4 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr4_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr5 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr5_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr6 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr6_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr7 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr7_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr8 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr8_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr9 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr9_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr10 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr10_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr11 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr11_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr12 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr12_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr13 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr13_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    'CMD bin online Ne=100 snr14 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr14_Ne100000'.format(Nt, Nr, L), 'b-x', y_axis, True, True],
    # Additional simulations
    'CMD bin online Ne=10 snr8 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online10_snr8_Ne100000'.format(Nt, Nr, L), 'c-x', y_axis, True, True],
    'CMD bin online Ne=100 snr8 Nit1000000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr8_Ne1000000'.format(Nt, Nr, L), 'y-x', y_axis, True, True],
    'CMD bin online Ne=1000 snr8 Nit100000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online1000_snr8_Ne100000'.format(Nt, Nr, L), 'g-x', y_axis, True, True],
    # 'CMD bin online Ne=100 snr11 Nit100000 spdef': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr11_Ne100000_spdef'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    # 'CMD bin online Ne=100 snr11 Nit100000 splin': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr11_Ne100000_splin'.format(Nt, Nr, L), 'c-x', y_axis, True, True],
    'CMD bin online Ne=100 snr11 Nit1000000': ['RES_CMDNet_'+mod+'_{}_{}_{}_binary_online100_snr11_Ne1000000'.format(Nt, Nr, L), 'y-x', y_axis, True, True],
    # Online learning DNN
    'DNN online Ne=100 snr1 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr1_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr2 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr2_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr3 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr3_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr4 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr4_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr5 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr5_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr6 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr6_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr7 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr7_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr8 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr8_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr8 Nit100000 NL2 NW512 MSE': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr8_Ne100000_NL2NW512_mse'.format(Nt, Nr, L), 'm->', y_axis, True, True],
    'DNN online Ne=100 snr9 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr9_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr10 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr10_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr11 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr11_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr12 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr12_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr13 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr13_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    'DNN online Ne=100 snr14 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100_snr14_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    # Different training settings: online iterations + different architecture + ...
    # 'DNN online Ne=1000 snr1 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr1_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    # Ne=1000
    'DNN online Ne=1000 snr1 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr1_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr2 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr2_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr3 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr3_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr4 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr4_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr5 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr5_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr6 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr6_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr7 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr7_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr8 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr8_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr9 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr9_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr10 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr10_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr11 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr11_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr12 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr12_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr13 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr13_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr14 Nit100000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr14_Ne100000_NL2NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr8 Nit100000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr8_Ne100000_NL6NW512'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    'DNN online Ne=1000 snr8 Nit100000 NL6 NW512 MSE': ['RES_DNN_'+mod+'_{}_{}_{}_online1000_snr8_Ne100000_NL6NW512_mse'.format(Nt, Nr, L), 'm-+', y_axis, True, True],
    # Ne=10000
    'DNN online Ne=10000 snr1 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr1_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr2 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr2_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr3 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr3_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr4 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr4_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr5 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr5_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr6 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr6_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr7 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr7_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr8 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr8_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr8 Nit1000000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr8_Ne1000000_NL2NW512'.format(Nt, Nr, L), 'm-<', y_axis, True, True],
    # 'DNN online Ne=10000 snr8 Nit1000000 NL6 NW512 lrs': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr8_Ne1000000_NL6NW512_lrs'.format(Nt, Nr, L), 'm->', y_axis, True, True],
    'DNN online Ne=10000 snr9 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr9_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr10 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr10_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr11 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr11_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr12 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr12_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr13 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr13_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=10000 snr14 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online10000_snr14_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    # Ne=100000
    'DNN online Ne=100000 snr1 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr1_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr2 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr2_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr3 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr3_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr4 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr4_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr5 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr5_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr6 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr6_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr7 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr7_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr8 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr8_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr8 Nit1000000 NL2 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr8_Ne1000000_NL2NW512'.format(Nt, Nr, L), 'm-x', y_axis, True, True],
    'DNN online Ne=100000 snr9 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr9_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr10 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr10_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr11 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr11_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr12 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr12_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr13 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr13_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
    'DNN online Ne=100000 snr14 Nit1000000 NL6 NW512': ['RES_DNN_'+mod+'_{}_{}_{}_online100000_snr14_Ne1000000_NL6NW512'.format(Nt, Nr, L), 'm-o', y_axis, True, True],
}

# Choose table of simulation results here
ml_methods = bin_comparison  # cmdpar, one_ring2, sgd, online, code_ldpc, layer, train

# Processing
modobj = com.modulation(mod)
M0 = modobj.M
if x_axis == 'snr':
    # EbN0
    snr_shift = 10 * np.log10(2 * np.log2(M0))  # Always scale by 2 (N0/2)...
else:
    # SNR
    snr_shift = 0


# Performance curves
plt.figure(1)

# Load SIC and SD data from Dirk Wübben's Matlab implementation
fn_ext = ''  # '_rxrho07', '_OneRing20_120', '_rho05', '_rho0.5'
path_ber = os.path.join(path, 'BER_Files')
# file = os.path.join(path, 'BER_LIN_' + mod + '_{}_{}'.format(Nt, Nr) + fn_ext + '.mat')
# if os.path.isfile(file):
#     ber_sic = sio.loadmat(file)['BER'][0,0][['MF_LD', 'ZF_LD', 'MMSE_LD', 'EbN0']]
#     plt.semilogy(ber_sic[-1][0,:] + snr_shift, ber_sic[0][0,:], 'b--', label='MF')
#     plt.semilogy(ber_sic[-1][0,:] + snr_shift, ber_sic[1][0,:], 'b-<', label='ZF')
#     plt.semilogy(ber_sic[-1][0,:] + snr_shift, ber_sic[2][0,:], 'b-*', label='MMSE')
fn = 'BER_SIC_' + mod + '_{}_{}'.format(Nt, Nr) + fn_ext + '.mat'
pathfn = [os.path.join(path, fn), os.path.join(path_ber, fn)]
for file in pathfn:
    if os.path.isfile(file):
        ber_sic = sio.loadmat(file)['BER'][0, 0][[
            'ZF_SIC', 'MMSE_SIC', 'EbN0']]
        # plt.semilogy(ber_sic[-1][0,:] + snr_shift, ber_sic[0][0,:], 'y-<', label='ZF SIC')
        plt.semilogy(ber_sic[-1][0, :] + snr_shift,
                     ber_sic[1][0, :], 'y-*', label='MMSE SIC')
fn = 'BER_SD_' + mod + '_{}_{}'.format(Nt, Nr) + fn_ext + '.mat'
pathfn = [os.path.join(path, fn), os.path.join(path_ber, fn)]
for file in pathfn:
    if os.path.isfile(file):
        try:
            ber_sic = sio.loadmat(file)['BER'][0, 0][['SD', 'EbN0']]
            # überschreiben, falls andere Datei vorhanden
            plt.semilogy(ber_sic[-1][0, :] + snr_shift, ber_sic[0]
                         [0, :], 'k-o', label='Sphere Detector')
        except KeyError:
            ber_sic = sio.loadmat(file)['BER_SD']
            snr_sd = sio.loadmat(file)['SNR_SD']
            plt.semilogy(snr_sd + snr_shift, ber_sic,
                         'k-o', label='Sphere Detector')


# Loading files from table
res0 = 0
for algo, algo_set in ml_methods.items():
    if algo_set[-1]:
        pathfile = os.path.join(path, algo_set[0])
        res0 = load.load(pathfile, form='npz')
        if res0 is not None:
            res = res0
            if algo_set[2] in res:
                if mode == 1:
                    # Equalization + Channel coding
                    if mode_online and algo_set[-2] == True:
                        # Online learning
                        plt.semilogy(res[algo_set[2]][0], np.mean(
                            res[algo_set[3]]), algo_set[1], label=algo)
                    else:
                        plt.semilogy(
                            res[algo_set[2]], res[algo_set[3]], algo_set[1], label=algo)
                else:
                    # Only equalization
                    if mode_online and algo_set[-2] == True:
                        # Online learning
                        plt.semilogy(res[x_axis][0], np.mean(
                            res[algo_set[2]]), algo_set[1], label=algo)
                    else:
                        plt.semilogy(
                            res[x_axis], res[algo_set[2]], algo_set[1], label=algo)


# Calculate AWGN curve that approximates BER curves for antenna dimensions growing to infinity
snr_awgn = np.arange(-6 + snr_shift, 30 + snr_shift, 1)
if mod == 'QAM16':
    M = 16
    ber_awgn = 2 / np.log2(M) * (1 - 1 / np.sqrt(M)) * sp.erfc(
        np.sqrt(3 * np.log2(M) / (2 * (M - 1)) * mop.dbinv(snr_awgn)))
else:
    ber_awgn = 0.5 * sp.erfc(np.sqrt(mop.dbinv(snr_awgn)))
plt.semilogy(snr_awgn, ber_awgn, 'g-', label='AWGN')
# Additional interesting reference curves
# from scipy.stats import norm
# ber_ray = 0.5 * (1 - np.sqrt(mop.dbinv(snr_awgn) / (1 + mop.dbinv(snr_awgn))))
# ber_mlapp = 1 - norm.cdf(np.sqrt(mop.dbinv(snr_awgn)))
# plt.semilogy(snr_awgn, ber_ray, 'g--', label = 'Rayleigh fading')#, label = 'xyz')
# plt.semilogy(snr_awgn, ber_mlapp, 'g--', label = 'ML approx')#, label = 'xyz') # ?


# Plot options
plt.xlim(-6 + snr_shift, 30 + snr_shift)
# plt.xlim(-6 + snr_shift, 40 + snr_shift)
# plt.xlim(1, 18)
plt.ylim(10 ** -6, 1)
plt.grid(visible=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(visible=True, which='minor',
         color='#999999', linestyle='-', alpha=0.3)
plt.xlabel(x_axis)
plt.ylabel(y_axis)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

tplt.save("plots/MIMO_"+mod+"_{}x{}_{}".format(Nt, Nr, L) + fn_tikz + ".tikz")
