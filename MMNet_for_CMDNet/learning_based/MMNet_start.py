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

def run_ipython(cmd):
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is None:
            raise RuntimeError("Not in IPython")
        ip.run_line_magic('run', cmd)
    except Exception as e:
        raise RuntimeError("IPython execution requested but not available") from e


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    run_sim = 0
    jobname = 'MMNet_test'
    # Cluster options
    mem = 32000
    time = '30-00:00:00'
    mail = 'beck@ant.uni-bremen.de'
    node = 'fuchu'
    exclude = ''

    # Parameters
    Nt2 = 64
    Nr2 = 64
    Nt = int(Nt2 / 2)
    Nr = int(Nr2 / 2)
    L = Nt2                         # default: 10 vs. Nt2 (CMD)
    ebn0_min = 4
    # default: snr range [SER: 10^-2 - 10^-3] vs. [4, 27] vs. [4, 11]
    ebn0_max = 27
    lr = 1e-3                       # default: 1e-3 (Adam)
    batch_size = 500                # default: 500 vs. 5000 (DetNet) vs. 500 (CMD)
    # default: 10000 vs. 50000 (DetNet) vs. 100000 (CMD)
    train_it = 100000               # default: 100000
    M = 4                           # Number of symbols
    mod = 'QAM_{}'.format(M)        # modulation: QAM_4, QAM_16, QAM_64
    snr_shift = int(np.round(10 * np.log10(np.log2(M))))
    mmnet_shift = int(10 * np.log10(Nr2 / Nt2))
    snr_min = ebn0_min + snr_shift - mmnet_shift
    snr_max = ebn0_max + snr_shift - mmnet_shift
    test_batch_size = 10000         # 10000
    test_iteration_print = 500      # 500
    linear = 'MMNet_iid'            # linear: MMNet, MMNet_iid, lin_DetNet, OAMPNet
    denoiser = 'MMNet'              # denoiser: MMNet, DetNet, OAMPNet

    gpu_selected = 1                # Default: 0
    complex_system = 1              # 1
    angularspread = 0               # 20, 10
    cell_sector = 120               # 120
    fn_ext = '_test'                # '_OneRing10_120_snr4_27'

    # Save in CMDNet root folder
    # save_dir = ''
    save_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    save_dir = os.path.abspath(save_dir)

    cmd_str = 'offlineTraining_CMDNet.py --x-size {} --y-size {} --snr-min {} --snr-max {} --layers {} -lr {} --batch-size {} --train-iterations {} --mod {}  --test-batch-size {} --linear {}  --denoiser {}  --test-every {} --gpu {} --complex {} --angularspread {} --cellsector {} --filename_extension {} --save_directory {}'.format(
        Nt, Nr, snr_min, snr_max, L, lr, batch_size, train_it, mod, test_batch_size, linear, denoiser, test_iteration_print, gpu_selected, complex_system, angularspread, cell_sector, fn_ext, save_dir)
    if run_sim == 1 or run_sim == 2:
        # Write simulation file
        conda_env = 'ml1'
        bash_cmd = 'python -u ' + \
            cmd_str + ' > ' + jobname + '.out 2> ' + jobname + '.err'
        if run_sim == 1:
            # Run file in background on computer
            bash_cmd = 'nohup ' + bash_cmd
        mysc = '#!/bin/bash\nsource ~/.bashrc\nconda activate {}\n'.format(
            conda_env) + bash_cmd
        fn = 'MMNet_sim.sh'
        file = open(fn, 'w')
        file.write(mysc)
        file.close()


    if run_sim == 1:
        # Run simulation in background
        os.system('./' + fn)
    elif run_sim == 2:
        # Run simulation on cluster
        bash_cmd2 = 'sbatch --mem={} --time={} --job-name={} --partition=wiss --nodelist={} --exclude={} --mail-user={} --mail-type=ALL --gpus=1 {}'.format(
            mem, time, jobname, node, exclude, mail, fn)
        os.system(bash_cmd2)
    else:
        # Run interactive inside IPython / Jupyter
        run_ipython(cmd_str)
        # Run interactive in spyder
        # runfile(cmd_str)
