#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:22:21 2019

@author: beck
"""
import numpy as np
import os
# ORIGINAL:
# python offlineTraining.py  --x-size 16 --y-size 64 --snr-min 2 --snr-max 7 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 10000 --mod QAM_4  --test-batch-size 5000 --linear MMNet_iid --denoiser MMNet --test-every 100
# python onlineTraining.py  --x-size 32 --y-size 64 --snr-min 10 --snr-max 15 --layers 10 -lr 1e-3 --batch-size 500 --train-iterations 1000 --mod QAM_4  --test-batch-size 5000 --linear MMNet  --denoiser MMNet --data --channels-dir path/to/channels --output-dir path/to/save/results

run_sim = 1
jobname = 'MN_OR3'
# Cluster options
mem = 32000
time = '30-00:00:00'
mail = 'beck@ant.uni-bremen.de'

# Parameters
Nt2 = 64
Nr2 = 128
Nt = int(Nt2 / 2)
Nr = int(Nr2 / 2)
L = Nt2                         # default: 10 vs. Nt2 (CMD)
ebn0_min = 4
# default: snr range [SER: 10^-2 - 10^-3] vs. [4, 27] vs. [4, 11]
ebn0_max = 27
lr = 1e-3                       # default: 1e-3 (Adam)
batch_size = 500                # default: 500 vs. 5000 (DetNet) vs. 500 (CMD)
# default: 10000 vs. 50000 (DetNet) vs. 100000 (CMD)
train_it = 100000
M = 4                           # Number of symbols
mod = 'QAM_{}'.format(M)        # modulation: QAM_4, QAM_16, QAM_64
snr_shift = int(np.round(10 * np.log10(np.log2(M))))
mmnet_shift = int(10 * np.log10(Nr2 / Nt2))
snr_min = ebn0_min + snr_shift - mmnet_shift
snr_max = ebn0_max + snr_shift - mmnet_shift
test_batch_size = 10000
test_int = 500
linear = 'MMNet_iid'            # linear: MMNet, MMNet_iid, lin_DetNet, OAMPNet
denoiser = 'MMNet'              # denoiser: MMNet, DetNet, OAMPNet


cmd_str = 'offlineTraining_CMDNet.py --x-size {} --y-size {} --snr-min {} --snr-max {} --layers {} -lr {} --batch-size {} --train-iterations {} --mod {}  --test-batch-size {} --linear {}  --denoiser {}  --test-every {}'.format(
    Nt, Nr, snr_min, snr_max, L, lr, batch_size, train_it, mod, test_batch_size, linear, denoiser, test_int)
conda_env = 'machine_learning'
bash_cmd = 'nohup python -u learning_based/' + \
    cmd_str + ' > ' + jobname + '.out 2> ' + jobname + '.err'
mysc = '#!/bin/bash\nsource ~/.bashrc\nconda activate {}\n'.format(
    conda_env) + bash_cmd
fn = 'mysim.sh'
file = open(fn, 'w')
file.write(mysc)
file.close()

if run_sim == 1:
    # Run simulation
    os.system('./' + fn)
elif run_sim == 2:
    # Run simulation on cluster
    bash_cmd2 = 'sbatch --mem={} --time={} --job-name={} --partition=wiss --mail-user={} --mail-type=ALL {}'.format(
        mem, time, jobname, mail, fn)
    os.system(bash_cmd2)
else:
    # Run interactive in spyder
    runfile(cmd_str)
