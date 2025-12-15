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

import os
import tikzplotlib as tplt
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np

# Own packages
import utilities.my_functions as mf
import utilities.my_training as mt

# Settings
fn_ext = ''
Nt = 64
Nr = 64
L = 64
mod = 'QPSK'
y_axis = 'params'       # val_loss, train_loss, params
it = -1                 # int(100000/ 100) # iteration for params:0, -1
x_axis = 'epoch'        # epoch
path = os.path.join('curves', mod, '{}x{}'.format(Nt, Nr))
load = mf.savemodule()

online = {  # 'CMD bin online': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_online100_snr8'.format(Nt, Nr, L), 'b-', True],
    'CMD bin': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'g--', True],
    # 'CMD bin tau0.01': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'g--', True],
    'CMD bin online 100': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_online100_snr8_2'.format(Nt, Nr, L), 'b-', True],
    # 'CMD bin online 1000': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_online1000_snr8_2'.format(Nt, Nr, L), 'r-', True],
    # 'CMD bin online test': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_test'.format(Nt, Nr, L), 'k-', True],
}

cross_entr = {'CMD bin tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'b-', True],
              # 'CMD CE1': ['trainhist_CMD_'+mod+'_{}_{}_{}_softmax_tau0.1'.format(Nt, Nr, L), 'k-', True],
              # 'CMD CE2': ['trainhist_CMD_'+mod+'_{}_{}_{}_softmax_tau0.1_2'.format(Nt, Nr, L), 'y-', True],
              # 'CMD': ['trainhist_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'r-', True],
              'CMD bin mseloss': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_mseloss'.format(Nt, Nr, L), 'c--', True],
              }

value_sp = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 128 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_128_binary_tau0.1'.format(Nt, Nr), 'r--', True],
    'CMD bin 64 tau0.01': ['trainhist_CMD_'+mod+'_{}_{}_64_binary'.format(Nt, Nr), 'r-', True],
    'CMD bin 64 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_64_binary_tau0.1'.format(Nt, Nr), 'k-', True],
    # 'CMD bin 16': ['trainhist_CMD_'+mod+'_{}_{}_16_binary'.format(Nt, Nr), 'y-', True],
    # 'CMD bin 16 tau0.01': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_correct'.format(Nt, Nr), 'g-', True],
    # 'CMD bin 16 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_correct3'.format(Nt, Nr), 'k-', True],
    # 'CMD bin 16 tau0.075': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'k-', True],
    # 'CMD binc': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_comb'.format(Nt, Nr, L), 'y-', True],
}

type_sp = {  # 'CMD def': ['trainhist_CMD_'+mod+'_{}_{}_{}_tau0.1'.format(Nt, Nr, L), 'b-', True],
    # 'CMD bin def 0': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-', True],
    'CMD bin spconst': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_spconst'.format(Nt, Nr, L), 'k-', True],
    'CMD bin splin': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'g-', True],
    'CMD bin def -1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-', True],
}

qam16 = {'CMD snr10_33': ['trainhist_CMD_'+mod+'_{}_{}_{}_snr10_33'.format(Nt, Nr, L), 'b-', True],
         'CMD convex': ['trainhist_CMD_'+mod+'_{}_{}_{}_convex'.format(Nt, Nr, L), 'r-', True],
         # 'CMD': ['trainhist_CMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'g-', True],
         # 'CMD tauscale': ['trainhist_CMD_'+mod+'_{}_{}_{}_tauscale'.format(Nt, Nr, L), 'k-', True],
         }

jparams = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 16 tau0.075': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'k-', True],
    'CMD bin 64 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_64_binary_tau0.1'.format(Nt, Nr), 'r-', True],
    'CMD bin 64 splin': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'b-', True],
}

cmdpar = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 16 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r-', True],
    'CMDpar 16 1': ['trainhist_CMDpar_'+mod+'_{}_{}_16_binary_test'.format(Nt, Nr), 'y-', y_axis, True],
    'CMDpar 16 2': ['trainhist_CMDpar_'+mod+'_{}_{}_16_binary_test2'.format(Nt, Nr), 'g-', y_axis, True],
    # 'CMDpar 64': ['trainhist_CMDpar_'+mod+'_{}_{}_64_binary_test2'.format(Nt, Nr), 'b-', y_axis, True],
}

hypercmd = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 16 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_16_binary_tau0.075'.format(Nt, Nr), 'r-', True],
    'CMD bin 16x16': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r--', y_axis, True],
    'HyperCMD': ['trainhist_HyperCMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'b--', y_axis, True],
    # 'HyperCMD 16': ['trainhist_HyperCMD_'+mod+'_{}_{}_16'.format(Nt, Nr, L), 'b-', y_axis, True],
    # 'HyperCMD 8': ['trainhist_HyperCMD_'+mod+'_{}_{}_8_binary'.format(Nt, Nr, L), 'g--', y_axis, True],
    # 'HyperCMD': ['trainhist_HyperCMD_'+mod+'_{}_{}_{}'.format(Nt, Nr, L), 'b-o', y_axis, True],
}

corr = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 64': ['trainhist_CMD_'+mod+'_{}_64_{}_binary_splin_tau0.1'.format(Nt, L), 'b-', True],
    'CMD bin': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary'.format(Nt, Nr, L), 'r-', True],
    'CMD bin corr': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_OneRing20_120'.format(Nt, Nr, L), 'y-', True],
}

sgd = {  # 'Tag': ['data name', 'color in plot', on/off],
    'CMD bin 64 tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_tau0.1'.format(Nt, Nr, L), 'r-', True],
    'CMD bin splin tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_splin_tau0.1'.format(Nt, Nr, L), 'b-', True],
    # 'CMD bin splin tau0.01': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_splin'.format(Nt, Nr, L), 'b--', True],
    'CMD bin sgd Nb32 splin tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_sgdNb32_splin'.format(Nt, Nr, L), 'y-', True],
    'CMD bin sgdmom Nb32 splin nesterov': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_splin_sgdmomNb32'.format(Nt, Nr, L), 'g-', True],
    'CMD bin sgdmom Nb500 tau0.1 lr': ['trainhist_CMDNet_'+mod+'_{}_{}_{}_binary_sgdmomNb500_tau0.1_lr'.format(Nt, Nr, L), 'g--', True],
    # 'CMD bin sgd Nb500 splin tau0.1': ['trainhist_CMD_'+mod+'_{}_{}_{}_binary_sgd_splin'.format(Nt, Nr, L), 'y-', True],
}

ml_methods = sgd

# Training stats

plt.figure(1)
train_hist = mt.TrainingHistory()
for algo, algo_set in ml_methods.items():
    if algo_set[-1]:
        pathfile = os.path.join(path, algo_set[0])
        res = load.load(pathfile, form='npz')
        train_hist.dict2obj(res)
        if res is not None:
            if y_axis == 'params':
                plt.figure(1)
                # plt.plot(res['params'][0][0], algo_set[1], label=algo)
                # it = algo_set[-2]
                plt.plot(res[y_axis][it][0], algo_set[1], label=algo)
                plt.figure(2)
                # # plt.plot(1 / res['params'][0][1], algo_set[1], label=algo)
                plt.plot(1 / res[y_axis][it][1], algo_set[1], label=algo)
            else:
                plt.semilogy(res[x_axis], res[y_axis], algo_set[1], label=algo)
                train_hist.printh()

if y_axis == 'params':
    plt.figure(1)
    plt.ylim(min(res[y_axis][it][0]), max(res[y_axis][it][0]))
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.xlabel('iteration')
    plt.ylabel('delta')
    plt.legend()

    plt.figure(2)
    plt.ylim(min(1 / res[y_axis][it][1]), max(1 / res[y_axis][it][1]))
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.xlabel('iteration')
    plt.ylabel('tau')
    plt.legend()
else:
    plt.xlim(min(res[x_axis]), max(res[x_axis]))
    plt.ylim(min(res[y_axis]), max(res[y_axis]))
    plt.grid(b=True, which='major', color='#666666', linestyle='-')
    plt.minorticks_on()
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.3)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.legend()


# plt.show() # comment if you want to save with tplt
tplt.save("plots/trainhist_MIMO_"+mod +
          "_{}x{}_{}".format(Nt, Nr, L) + fn_ext + ".tikz")

# val_loss evaluation for online training
# val_loss = res[y_axis][1:21301]
# val_loss2 = val_loss.reshape((-1, 100))
# plt.figure(2)
# #val_loss = res[y_axis][1:] #:43001
# #val_loss2 = val_loss.reshape((-1, 1000))
# val_loss2 = np.mean(val_loss2, axis = 0)
# plt.plot(val_loss2)
