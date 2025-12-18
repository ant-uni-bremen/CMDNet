#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:05:19 2019

@author: beck
"""

import sys                                  # NOQA
# Include current folder, where start simulation script and packages are
sys.path.append('.')                        # NOQA
# Include parent folder, where own packages are
sys.path.append('..')                       # NOQA

# LOADED PACKAGES
import os

# Python packages
import numpy as np
import tensorflow as tf
import yaml
# Tensorflow/Keras packages
# from tensorflow import keras
from tensorflow.keras import backend as KB

# Own packages
import utilities.my_communications as com
import utilities.my_equalizer as equ
import utilities.my_functions as mf
import utilities.my_math_operations as mop
import utilities.my_mimo_channel as mch
import utilities.my_training_tf1 as mt
import cmdnet_extensions as cmd_extensions
import cmdnet


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Set working directory to script directory
    # Path of script being executed
    path0 = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path0)

    # Load settings
    # sim_settings.yaml, sim_settings_default.yaml, sim_settings_code.yaml, sim_settings_llr.yaml, sim_settings_online.yaml, sim_settings_onlineDNN.yaml, sim_settings_deeq.yaml
    settings_file = "sim_settings.yaml"

    with open(settings_file, 'r', encoding='UTF8') as stream:
        set_dict = yaml.safe_load(stream)
    # Settings file already loaded? Can be changed then...
    print('"' + settings_file + '" loaded.')
    load_set = set_dict['load_set']
    # 0/1: BER evaluation, 2: LLR evaluation
    llr_mode = load_set['sim']
    sim_set = set_dict['sim_set']
    code_set = set_dict['code_set']
    train_set = set_dict['train_set']

    # Initialization
    KB.clear_session()                  # clearing graphs
    # np.random.seed(0)                 # predictable random numbers for debugging
    np.random.seed()                    # random seed in every run
    # computation accuracy: 'float16', 'float32', or 'float64'
    KB.set_floatx(load_set['prec'])
    if load_set['gpu'] == 0:
        mt.tf_enable_gpu(0, 8)          # disable GPU
        # mt.tf_enable_gpu(1, 1)        # enable GPU / default

    # Create simulation objects
    mod = com.modulation(sim_set['Mod'])
    # Parameters: Nt, Nr, L, mod, N_batch, EbN0
    sim_params = mf.simulation_parameters(
        sim_set['Nt'], sim_set['Nr'], sim_set['L'], mod, sim_set['Nbatch'], sim_set['ebn0_range'], sim_set['rho'])
    # Optional input # TODO: Dimensions fit w/o input?
    sim_params.mod.alpha = 1 / sim_params.mod.M * \
        np.ones((sim_params.Nt, sim_params.mod.M))
    sim_params.snr_gridcalc(sim_set['snr_grid'])
    train_params = mf.simulation_parameters(sim_params.Nt, sim_params.Nr, sim_params.L,
                                            sim_params.mod, train_set['batch_size'], train_set['ebn0_range'], sim_params.rho)

    # Create save object
    saveobj = mf.savemodule(load_set['sv_ext'])
    # Load code matrices
    codeon = (code_set['code'] != 'uncoded')
    if codeon:
        code = saveobj.load(os.path.join(code_set['path'], code_set['code']))
        G = code['G']
        H = code['H']
    # Create filename object -> for training without code parameters (TODO: maybe train with code)
    fn = mf.filename_module(
        load_set['fn_train'], sim_set['path'], sim_set['algo'], sim_set['fn_ext'], sim_set)

    # TRAINING/ALGORITHM INITIALIZATION
    # Load DetNet
    if sim_set['algo'] == 'DetNet':
        detnet = cmdnet.DetNetGraph(train_params, sim_set['fn_ext'])

    if llr_mode == 2:
        # LLR comparison
        # Load DetNet [fixed comparison to this version of DetNet]
        # if sim_set['algo'] == 'DetNet':
        train_ebn0_detnet = [4, 11]
        fn_ext_detnet = '_defsnr'
        train_params2 = mf.simulation_parameters(
            sim_params.Nt, sim_params.Nr, sim_params.L, sim_params.mod, train_set['batch_size'], train_ebn0_detnet, sim_params.rho)
        detnet_llr = cmdnet.DetNetGraph(train_params2, fn_ext_detnet)

    # Load MMNet or OAMPNet
    if sim_set['algo'] in ['MMNet', 'OAMPNet']:
        mmnet = cmdnet.MMNetGraph(
            train_params, sim_set['algo'], sim_set['fn_ext'])

    # CMDNet/CMD
    if sim_set['algo'] == 'CMD_fixed':
        # Choose parameters / starting point
        if load_set['train'] == 1:
            fn2 = mf.filename_module(
                'trainhist_', 'curves', 'CMD', '_binary_splin', sim_set)
            train_hist = mt.TrainingHistory()
            train_hist.filename = fn2.pathfile
            train_hist.dict2obj(saveobj.load(train_hist.filename))
            # [delta0, taui0], _ = train_hist.sel_best_weights()
            [delta0, taui0] = train_hist.params[-1]
        else:
            [delta0, taui0] = cmdnet.cmd_initpar(
                sim_params.mod.M, sim_params.L, train_set['start_point'])

    cmd_algos = ['CMD', 'CMDNet', 'HyperCMD', 'CMDpar']     # CMDNet approaches

    if sim_set['algo'] in cmd_algos + ['DNN']:
        # Prepare data
        data_gen = mch.dataset_gen(
            train_params, soft=train_set['soft'], online=train_set['online'])
        data_gen.N_bval = train_set['val_size']
        data_gen.data_val = data_gen.gen_mimodata(data_gen.N_bval).copy()

        # Online learning: Check if last online learning iteration equals an iteration checkpoint -> last iteration needs to be saved
        if (train_set['online'] != 0) and (train_set['online'] % train_set['it_checkp'] != 0):
            train_set['it_checkp'] = train_set['online']

        # Initialize optimizer
        if train_set['lr_dyn']['mode'] == 2:
            # Dynamic learning rate schedule, piecwise constant decay
            # DetNet FullyCon: initial lr=0.0008, then decay by factor 0.97 after 1000 iterations, 1e6 iterations in total
            lrs = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                train_set['lr_dyn']['boundaries'], train_set['lr_dyn']['values'])
        else:
            # Default learning rate
            lrs = None
        opt = cmdnet.optimizer_select(opt_name=train_set['opt'], lrs=lrs)

    if sim_set['algo'] in cmd_algos:
        # Choose parameters / starting point
        if sim_set['algo'] == 'CMDpar':
            [delta0, taui0] = cmdnet.cmd_initpar(
                sim_params.mod.M, sim_params.L, train_set['start_point'], k=3)
        else:
            [delta0, taui0] = cmdnet.cmd_initpar(
                sim_params.mod.M, sim_params.L, train_set['start_point'])

        # Initialization
        train_hist = mt.TrainingHistory()
        train_hist.filename = fn.pathfile  # Set path for checkpoint saves
        # Load results if already trained
        if load_set['train'] == 1:
            train_hist.dict2obj(saveobj.load(train_hist.filename))

        # Create graph
        if sim_set['algo'] == 'HyperCMD':
            CMDgraph = cmd_extensions.HyperCMDGraph(train_params, delta0, taui0, soft=train_set['soft'],
                                                    multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        elif sim_set['algo'] == 'CMDpar':
            CMDgraph = cmd_extensions.CMDparGraph(train_params, delta0, taui0, soft=train_set['soft'],
                                                  multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        else:
            CMDgraph = cmdnet.CMDGraph(train_params, delta0, taui0, soft=train_set['soft'],
                                       multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        # Train
        train_hist = cmdnet.train(train_set['Nepoch'], train_hist, CMDgraph.train_inputs, CMDgraph.loss,
                                  CMDgraph.params, opt, data_gen, saveobj,
                                  it_checkp=train_set['it_checkp'], sv_checkp=train_set['sv_checkp'],
                                  sel_best_weights=train_set['sel_bweights'], esteps=train_set['esteps'],
                                  lr_dyn=train_set['lr_dyn']['mode'])

    # TEST (with or without code)
    if codeon:
        fn = mf.filename_module(
            load_set['fn_sim'], sim_set['path'], sim_set['algo'], sim_set['fn_ext'], sim_set, code_set)
        dataobj = mch.dataset_gen(
            sim_params, soft=0, code=1, G=G, arch=code_set['arch'])
        R_c = G.shape[0] / G.shape[-1]
        sel_crit = 'cber'
    else:
        fn = mf.filename_module(
            load_set['fn_sim'], sim_set['path'], sim_set['algo'], sim_set['fn_ext'], sim_set)
        dataobj = mch.dataset_gen(sim_params, soft=0)
        R_c = 1
        sel_crit = 'ber'
    fn.generate_pathfile_MIMO()
    # Save settings
    with open(os.path.join('settings', fn.filename + '.yaml'), 'w', encoding='utf8') as outfile:
        yaml.dump(set_dict, outfile, default_flow_style=False)
    # Stopping criterion in simulations
    it_max = int(sim_set['sim_prec'] ** -1 * sim_set['Nerr_min'] / (R_c * sim_params.N_batch *
                 sim_params.Nt * np.log2(sim_params.mod.M)))   # maximum number of iterations

    # Load results if already simulated
    perf_meas = mf.performance_measures(sim_set['Nerr_min'], it_max, sel_crit)
    if load_set['sim'] == 1:
        perf_meas.load_results(saveobj.load(fn.pathfile))

    if llr_mode != 2:
        if train_set['online'] == 0:
            # MAIN SCRIPT: BER evaluation with random channel and noise realizations
            for ii, snr in enumerate(sim_params.SNR):
                if snr not in perf_meas.SNR:
                    while perf_meas.stop_crit():    # Simulate until 1000 errors or stop after it_max iterations
                        # Generate test data
                        dataobj.sim_par.SNR_range = [snr, snr]
                        dataobj()

                        # Test algorithms: Equalization

                        # 0. ML detector
                        if sim_set['algo'] == 'MAP':
                            s, fr, _ = equ.ml_detector(
                                dataobj.data[0], dataobj.data[1], sim_params.mod.m, res=load_set['prec'])

                        # 1. MMSE solution
                        if sim_set['algo'] == 'MMSE':
                            s, Phi_ee = equ.mmse(dataobj.data[0:3])
                            fr = equ.lin_det_soft(
                                s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

                        # 2. LS solution
                        if sim_set['algo'] == 'LS':
                            s, Phi_ee = equ.ls_sol(dataobj.data[0:3])
                            fr = equ.lin_det_soft(
                                s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

                        # 3. Matched Filter
                        if sim_set['algo'] == 'MF':
                            s, Phi_ee = equ.matched_filter(dataobj.data[0:3])
                            fr = equ.lin_det_soft(
                                s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

                        # 4. MFVI (sequential or parallel)
                        if sim_set['algo'] == 'MFVIseq':
                            s, fr = equ.mfvi(
                                dataobj.data[0:3], sim_params.mod, sim_params.L, seq=1, binopt=sim_set['binopt'])
                        if sim_set['algo'] == 'MFVIpar':
                            s, fr = equ.mfvi(
                                dataobj.data[0:3], sim_params.mod, sim_params.L, seq=0, binopt=sim_set['binopt'])

                        # 5. AMP
                        if sim_set['algo'] == 'AMP':
                            s, fr = equ.AMP(
                                dataobj.data[0:3], sim_params.mod, sim_params.L, binopt=sim_set['binopt'])

                        # 6. CMD Unfolded: CMDNet / HyperCMD
                        if sim_set['algo'] in cmd_algos:
                            fr, s = CMDgraph.predict(dataobj.data[0:3])

                        # 7. CMD Detection
                        if sim_set['algo'] == 'CMD_fixed':
                            fr, s = equ.np_CMD(
                                dataobj.data[0:3], sim_params.mod, sim_params.L, delta0, taui0, binopt=sim_set['binopt'])

                        # 8. DetNet loaded
                        if sim_set['algo'] == 'DetNet':
                            a = np.sqrt(3 / (sim_params.mod.M ** 2 - 1))
                            # Method 1
                            HH = mop.batch_dot(np.transpose(
                                dataobj.data[1], (0, 2, 1)), dataobj.data[1])  # * sim_set['Nr']
                            # conversion since m = [-3,-1,3,1] * a in contrast to m = [-3,-1,1,3] in DetNet script
                            # * np.sqrt(sim_set['Nr'])
                            yH = mop.batch_dot(
                                dataobj.data[0], dataobj.data[1]) / a
                            fr, s = detnet.predict([yH, HH])
                            s = s * a
                            # fr = keras.utils.to_categorical(np.argmin(np.abs(s[:, :, np.newaxis] - sim_params.mod.m[np.newaxis, np.newaxis, :]) ** 2, axis = -1), num_classes = sim_params.mod.M)

                        # 9. MMNet / 10. OAMPNet loaded
                        if sim_set['algo'] in ['MMNet', 'OAMPNet']:
                            mmnet_shift = 10 * \
                                np.log10(train_params.Nr / train_params.Nt)
                            fr, z, s, x, _ = mmnet.predict(
                                [snr - mmnet_shift, snr - mmnet_shift, sim_params.N_batch, dataobj.data[1]])
                            # Extract data from tf graph
                            dataobj.z = z
                            dataobj.x = x

                        # 11. SDR
                        if sim_set['algo'] == 'SDR':
                            fr, s = equ.sdrSolver(
                                dataobj.data[1], dataobj.data[0], sim_params.mod.m)

                        # Decoding
                        if codeon:
                            [bp_out, cr, ur] = com.mimoequ_decoding(
                                fr, dataobj, sim_params, code_set['arch'], H, code_set['dec'], code_set['it'])

                            perf_meas.cber_calc(dataobj.u, ur)
                            perf_meas.cfer_calc(dataobj.u, ur)

                        # Performance evaluation
                        perf_meas.eval(dataobj.z, fr, mod)
                        perf_meas.mse_calc(dataobj.x, s)
                        perf_meas.err_print()

                    # Save only if accuracy high enough after it_max iterations
                    [print_str, sv_flag] = perf_meas.err_saveit(
                        sim_params.SNR[ii], sim_params.EbN0[ii], sim_params.EbN0[ii] - 10 * np.log10(R_c))
                    print('{}, '.format(len(sim_params.SNR) - ii) + print_str)
                    if sv_flag:
                        # Save results to file
                        saveobj.save(fn.pathfile, perf_meas.results())

        else:
            # ONLINE TRAINING SCRIPT - Addon for PhD Thesis
            # Online training only with varying channel matrix
            if sim_set['algo'] == 'DNN':
                # 1. Initialize online training DNN, define Keras model
                dnnmimo, loss = cmdnet.mimo_dnn(
                    sim_par=sim_params, Nh=train_set['dnnwidth'], NL=train_set['dnndepth'], soft=train_set['soft'])
                # Define metric
                if train_set['soft'] == 1:
                    # Define custom symbol-to-accuracy metric for MSE symbol optimization
                    def sym2acc_metric(constellation):
                        def sym2err_metric(y_true, y_pred):
                            ''' Calculate accuracy metric from symbols using previously stored modulation constellation
                            '''
                            fr_pred = KB.argmin(KB.abs(KB.expand_dims(
                                y_pred, axis=-1) - constellation[np.newaxis, np.newaxis, :]) ** 2, axis=-1)
                            fr_true = KB.argmin(KB.abs(KB.expand_dims(
                                y_true, axis=-1) - constellation[np.newaxis, np.newaxis, :]) ** 2, axis=-1)
                            return 1 - KB.mean(KB.equal(fr_pred, fr_true))
                        return sym2err_metric
                    metrics = [sym2acc_metric(sim_params.mod.m)]
                else:
                    # Default with softmax outputs
                    metrics = [tf.keras.metrics.CategoricalAccuracy()]
                dnnmimo.compile(optimizer=opt, loss=loss, metrics=metrics)

            # Online learning preparations
            Ne_online = train_set['online']
            # Compute number of different channel realizations
            if sim_set['algo'] in cmd_algos:
                online_result_it = np.arange(0, len(train_hist.epoch))[(np.array(
                    train_hist.epoch) % Ne_online == 0) * (np.array(train_hist.epoch) != 0)]
            else:
                online_result_it = np.arange(
                    1,  train_set['Nepoch'] // Ne_online + 1)
            if load_set['sim'] == 1:
                # Restrict to simulations not included in the loaded results
                online_result_it = online_result_it[len(perf_meas.SNR):]
            # Minimum number of validation data iterations
            it_min = 10  # default: 10
            # Calculation and training just for one SNR: Fix SNR!!!
            ii_snr = 0
            snr = sim_params.SNR[ii_snr]
            dataobj.sim_par.SNR_range = [snr, snr]

            for ii, ii_par in enumerate(online_result_it):
                # Iterate through different channel matrices
                if sim_set['algo'] in cmd_algos:
                    # Load weights and corresponding channel matrices and SNRs saved in training history
                    train_hist.sel_weights(CMDgraph.params, ii_par)
                    # delta0, taui0 = train_hist.params[ii_par]
                    Hr, sigma = train_hist.add_params[ii_par - 1]
                    dataobj.Hr = Hr[np.newaxis, :, :]
                    dataobj.sigma = np.array(sigma)[np.newaxis]
                else:
                    # 2. Train low-complex DNN here for one channel realization and same SNR, NOTE: Also other models possible -> replace dnnmimo by model
                    # Shuffle new channel matrix and train with Keras fit function
                    if train_set['lr_dyn']['mode'] == 2 and ii != 0:
                        # Workaround: Reset learning rate schedule per realization by new model compilation
                        opt = cmdnet.optimizer_select(
                            opt_name=train_set['opt'], lrs=lrs)
                        dnnmimo.compile(
                            optimizer=opt, loss=loss, metrics=metrics)
                    N_onlinedata = Ne_online * train_set['batch_size']
                    # 100000 * 500 -> 23 GB for received signal, but 180-250 GB in total needed...
                    if N_onlinedata > 100000 * 500 or (N_onlinedata == 100000 * 500 and train_set['dnnverbose'] == 0) or train_set['esteps'] != 0:
                        # Early stopping, select best weights for online learing DNN, only w.r.t. epochs and thus data generator
                        if train_set['esteps'] == 0:
                            callbacks = []
                        else:
                            callbacks = [tf.keras.callbacks.EarlyStopping(
                                patience=train_set['esteps'], restore_best_weights=train_set['sel_bweights'])]
                        # If online training dataset size to big for system memory, use dataset generator per training iteration
                        # Measure in Gb: round(sys.getsizeof(data_gen.data[0]) / 1024 ** 3, 2)
                        # Reset online dataset generator to create new channel matrix
                        data_gen.online_it = 0
                        data_gen.gen_online_mimodata()
                        # Initialization for validation data + offset of 11 with data generator (reason unkown), so number of iterations needs to be increased
                        data_gen.online = Ne_online + 1 + 11
                        # data_gen_mimo = DataGenMIMO2()    # Easy debugging, just comment out and watch data_gen_mimo.it
                        history = dnnmimo.fit(cmdnet.data_gen_mimo(data_gen),
                                              batch_size=None,
                                              epochs=Ne_online,
                                              steps_per_epoch=1,
                                              # validation_split = 0.2,
                                              validation_data=(
                                                  data_gen.data_val[0], data_gen.data_val[-1]),
                                              # shuffle = False,
                                              # validation_steps = 1,
                                              # 0, 1, 2 (default), 'auto'
                                              verbose=train_set['dnnverbose'],
                                              callbacks=callbacks,
                                              )
                    else:
                        # otherwise generate dataset once and train on it
                        # Reset online dataset generator to create new channel matrix
                        data_gen.online_it = 0
                        data_gen.gen_online_mimodata(Nb=N_onlinedata)
                        history = dnnmimo.fit(x=data_gen.data[0],
                                              y=data_gen.data[-1],
                                              batch_size=train_set['batch_size'],
                                              epochs=1,
                                              # validation_split = 0.2,
                                              validation_data=(
                                                  data_gen.data_val[0], data_gen.data_val[-1]),
                                              shuffle=False,
                                              validation_steps=1,
                                              # 0, 1 (default for interactive), 2, 'auto'
                                              verbose=train_set['dnnverbose'],
                                              )
                    dataobj.Hr = data_gen.Hr
                    # Validation SNR can be different from training SNR data_gen.sigma
                    dataobj.sigma = mop.csigma(snr)

                # Simulate until 1000 errors or stop after it_max iterations + simulate minimum number of iterations it_min
                while perf_meas.stop_crit() or len(perf_meas.N_ber) < it_min:
                    # Generate test data
                    dataobj.gen_onlineval_mimodata()

                    # Test online learned algorithms for equalization

                    # 6. CMD Unfolded: CMDNet / HyperCMD
                    if sim_set['algo'] in cmd_algos:
                        fr, s = CMDgraph.predict(dataobj.data[0:3])

                    # 13. Online learning low-complex DNN (Fixed channel matrix + SNR)
                    if sim_set['algo'] == 'DNN':
                        # 3. Validate here for different noise realizations
                        pred_target = dnnmimo.predict(dataobj.data[0])
                        if train_set['soft'] == 1:
                            # Soft estimation with MSE
                            s = pred_target
                            fr = equ.konehot(np.argmin(np.abs(
                                s[:, :, np.newaxis] - sim_params.mod.m[np.newaxis, np.newaxis, :]) ** 2, axis=-1), num_classes=sim_params.mod.M)
                        else:
                            # Class probabilites hard output
                            fr = pred_target
                            s = np.sum(
                                fr * sim_params.mod.m[np.newaxis, np.newaxis, :], axis=-1)

                    # Decoding
                    if codeon:
                        [bp_out, cr, ur] = com.mimoequ_decoding(
                            fr, dataobj, sim_params, code_set['arch'], H, code_set['dec'], code_set['it'])

                        perf_meas.cber_calc(dataobj.u, ur)
                        perf_meas.cfer_calc(dataobj.u, ur)

                    # Performance evaluation
                    perf_meas.eval(dataobj.z, fr, mod)
                    perf_meas.mse_calc(dataobj.x, s)
                    perf_meas.err_print()

                # Save only if accuracy high enough after it_max iterations
                [print_str, sv_flag] = perf_meas.err_saveit(
                    sim_params.SNR[ii_snr], sim_params.EbN0[ii_snr], sim_params.EbN0[ii_snr] - 10 * np.log10(R_c))
                print('{}, '.format(len(online_result_it) - ii) + print_str)
                if sv_flag:
                    # Save results to file (sorting not necessary since only one SNR evaluated)
                    saveobj.save(fn.pathfile, perf_meas.results(sort=0))

    elif llr_mode == 2:
        # SPECIAL MODE: LLR OUTPUT EVALUATION
        import matplotlib.pyplot as plt
        import tikzplotlib as tplt

        # Specific LLR simulation settings
        N_llr = 10                      # Number of LLR batches
        snr = sim_params.SNR[0]         # LLR evaluation SNR
        # Histogram resolution, 'auto' (default)
        N_bins = 100
        llr_max = 40                    # Infinity LLRs are clipped to this value in the plot
        llr_thresh = 1000000            # LLR threshold considered to be infinite LLRs

        llr_list = []
        llr2_list = []
        cl_list = []
        dataobj.sim_par.SNR_range = [snr, snr]
        dataobj()
        H = np.repeat(
            dataobj.data[1][0, :, :][np.newaxis, :, :], dataobj.data[1].shape[0], axis=0)

        for ii in range(0, N_llr):
            # Generate test data
            dataobj()
            dataobj.data[1] = H
            dataobj.data[0] = com.awgn(mop.batch_dot(
                H, dataobj.x), dataobj.data[2][:, np.newaxis])

            # Test algorithms
            # Equalization

            # 0. ML detector
            if sim_set['algo'] == 'MAP':
                s, fr, _ = equ.ml_detector(
                    dataobj.data[0], dataobj.data[1], sim_params.mod.m, res=load_set['prec'])

            # 1. MMSE solution
            if sim_set['algo'] == 'MMSE':
                s, Phi_ee = equ.mmse(dataobj.data[0:3])
                fr = equ.lin_det_soft(
                    s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

            # 2. LS solution
            if sim_set['algo'] == 'LS':
                s, Phi_ee = equ.ls_sol(dataobj.data[0:3])
                fr = equ.lin_det_soft(
                    s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

            # 3. Matched Filter
            if sim_set['algo'] == 'MF':
                s, Phi_ee = equ.matched_filter(dataobj.data[0:3])
                fr = equ.lin_det_soft(
                    s, Phi_ee, sim_params.mod.m, sim_params.mod.alpha)

            # 4. MFVI (sequential or parallel)
            if sim_set['algo'] == 'MFVIseq':
                s, fr = equ.mfvi(
                    dataobj.data[0:3], sim_params.mod, sim_params.L, seq=1, binopt=sim_set['binopt'])
            if sim_set['algo'] == 'MFVIpar':
                s, fr = equ.mfvi(
                    dataobj.data[0:3], sim_params.mod, sim_params.L, seq=0, binopt=sim_set['binopt'])

            # 5. AMP
            if sim_set['algo'] == 'AMP':
                s, fr = equ.AMP(
                    dataobj.data[0:3], sim_params.mod, sim_params.L, binopt=sim_set['binopt'])

            # 6. CMD Unfolded: CMDNet
            if sim_set['algo'] in cmd_algos:
                fr, s = CMDgraph.predict(dataobj.data[0:3])

            # 7. CMD Detection
            if sim_set['algo'] == 'CMD_fixed':
                fr, s = equ.np_CMD(
                    dataobj.data[0:3], sim_params.mod, sim_params.L, delta0, taui0, binopt=sim_set['binopt'])

            # 8. DetNet loaded
            if sim_set['algo'] == 'DetNet':
                a = np.sqrt(3 / (sim_params.mod.M ** 2 - 1))
                # Method 1
                HH = mop.batch_dot(np.transpose(
                    dataobj.data[1], (0, 2, 1)), dataobj.data[1])  # * sim_set['Nr']
                # Conversion necessary since m = [-3,-1,3,1] * a in contrast to m = [-3,-1,1,3] in DetNet script
                # * np.sqrt(sim_set['Nr'])
                yH = mop.batch_dot(dataobj.data[0], dataobj.data[1]) / a
                fr, s = detnet.predict([yH, HH])
                s = s * a
                # fr = keras.utils.to_categorical(np.argmin(np.abs(s[:, :, np.newaxis] - sim_params.mod.m[np.newaxis, np.newaxis, :]) ** 2, axis = -1), num_classes = sim_params.mod.M)

            # 9. MMNet / 10. OAMPNet loaded
            if sim_set['algo'] == 'MMNet' or sim_set['algo'] == 'OAMPNet':
                mmnet_shift = 10 * np.log10(train_params.Nr / train_params.Nt)
                fr, z, s, x, _ = mmnet.predict(
                    [snr - mmnet_shift, snr - mmnet_shift, sim_params.N_batch, dataobj.data[1]])
                # Extract data from tf graph
                dataobj.z = z
                dataobj.x = x

            # 11. SDR
            if sim_set['algo'] == 'SDR':
                fr, s = equ.sdrSolver(
                    dataobj.data[1], dataobj.data[0], sim_params.mod.m)

            # 12. DetNet loaded [fixed comparison]
            # if sim_set['algo'] == 'DetNet':
            a = np.sqrt(3 / (sim_params.mod.M ** 2 - 1))
            # Method 1
            HH = mop.batch_dot(np.transpose(
                dataobj.data[1], (0, 2, 1)), dataobj.data[1])  # * sim_set['Nr']
            # Conversion necessary since m = [-3,-1,3,1] * a in contrast to m = [-3,-1,1,3] in DetNet script
            # * np.sqrt(sim_set['Nr'])
            yH = mop.batch_dot(dataobj.data[0], dataobj.data[1]) / a
            fr2, s2 = detnet_llr.predict([yH, HH])
            s2 = s2 * a

            # Decoding
            # TODO: a-posteriori, but extrinsic information (a-posteriori / a-priori) required?
            llr_c2, _ = com.symprob2llr(fr, sim_params.mod.M)
            llr_c2_2, _ = com.symprob2llr(fr2, sim_params.mod.M)
            # llr_c_perm = com.mimo_decoding(llr_c2, G.shape[-1], sim_params.Nt, sim_params.mod.M, code_set['arch'])
            # llr_c = dataobj.intleav.deinterleave(llr_c_perm)

            # Performance evaluation (LLR histogramm)
            cl_list.append(dataobj.cl[:, 0])
            llr_list.append(llr_c2[:, 0, :])
            llr2_list.append(llr_c2_2[:, 0, :])
            print('it: {}/{}'.format(N_llr - ii, N_llr))

        llr_list = np.array(llr_list).reshape((-1))
        llr2_list = np.array(llr2_list).reshape((-1))
        cl_list = np.array(cl_list).reshape((-1))
        # a, b = np.histogram(llr_list[cl_list == 0])
        print('Simulation ended.')

        # Create LLR Histograms
        llr_list0 = llr_list[cl_list == 0]
        llr_list0[np.abs(llr_list0) >= llr_thresh] = np.sign(
            llr_list0[np.abs(llr_list0) >= llr_thresh]) * llr_max
        llr2_list0 = llr2_list[cl_list == 0]
        llr2_list0[np.abs(llr2_list0) >= llr_thresh] = np.sign(
            llr2_list0[np.abs(llr2_list0) >= llr_thresh]) * llr_max
        hist1 = plt.hist(llr_list0, bins=N_bins, label=sim_set['algo'] + ' +1', weights=np.zeros_like(
            llr_list0) + 100. / llr_list0.size)  # Arguments are passed to np.histogram, bins='auto'
        hist2 = plt.hist(llr2_list0, bins=N_bins, label='DetNet +1',
                         weights=np.zeros_like(llr2_list0) + 100. / llr2_list0.size)
        # plt.title("Frequency LLRs x = +1")
        llr_list1 = llr_list[cl_list == 1]
        llr_list1[np.abs(llr_list1) >= llr_thresh] = np.sign(
            llr_list1[np.abs(llr_list1) >= llr_thresh]) * llr_max
        llr2_list1 = llr2_list[cl_list == 1]
        llr2_list1[np.abs(llr2_list1) >= llr_thresh] = np.sign(
            llr2_list1[np.abs(llr2_list1) >= llr_thresh]) * llr_max
        hist3 = plt.hist(llr_list1, bins=N_bins, label=sim_set['algo'] + ' -1', weights=np.zeros_like(
            llr_list1) + 100. / llr_list1.size)  # Arguments are passed to np.histogram, bins='auto'
        hist4 = plt.hist(llr2_list1, bins=N_bins, label='DetNet -1',
                         weights=np.zeros_like(llr2_list1) + 100. / llr2_list1.size)
        plt.title("Relative Frequency of LLRs")
        plt.legend()
        plt.ylim(0, np.max(hist3[0]) + 1)
        # plt.xlim(10, 20)
        fn_tikz = ''
        tplt.save("plots/MIMOLLRhistogram_" + mod.mod_name + "_{}x{}_{}".format(
            sim_params.Nt, sim_params.Nr, sim_params.L) + fn_tikz + ".tikz")
        # Statistics
        # It is well known that for Gaussian distributed noise, the LLRs also follow a Gaussian distribution (for a given value + 1 or −1 of the transmit signal) with mean and variance [5]
        # μ_L=4*EsN0 and σ_L^2=2*μ_L=8*EsN0
        channel_pow = np.sum(H[0, 0, :] ** 2)
        print('Channel power: {}'.format(channel_pow))
        print('Condition number: {}'.format(np.linalg.cond(H[0, :, :])))
        print('EsN0: {}'.format(
            10 ** (sim_params.EbN0[0] / 10) * np.log2(sim_params.mod.M) * channel_pow))
        print('LLR mean 0: {}'.format(np.mean(llr_list0)))
        print('LLR variance 0: {}'.format(np.var(llr_list0)))
        print('LLR mean 1: {}'.format(np.mean(llr_list1)))
        print('LLR variance 1: {}'.format(np.var(llr_list1)))

        # Save frequencies and histograms in dat-file
        # Bar data in one file
        # plt.hist: Returns values of histogram bins and bin edges
        # TikZ histogram example with bin edges: \addplot+[ybar interval] table{data.dat};
        # Append 0 as last histogram bin for compatibility with TikZ and ybar interval option
        np.savetxt("plots/data/histogram/MIMOLLRhistogram_" + sim_set['algo'] + "_" + mod.mod_name + "_{}x{}_{}".format(sim_params.Nt, sim_params.Nr, sim_params.L) + fn_tikz + ".dat",
                   np.c_[np.append(hist1[0], 0), hist1[1],
                         np.append(hist3[0], 0), hist3[1]],
                   )
        np.savetxt("plots/data/histogram/MIMOLLRhistogram_" + "DetNet" + "_" + mod.mod_name + "_{}x{}_{}".format(sim_params.Nt, sim_params.Nr, sim_params.L) + fn_tikz + ".dat",
                   np.c_[np.append(hist2[0], 0), hist2[1],
                         np.append(hist4[0], 0), hist4[1]],
                   )
        # Frequency saved -> too large for TikZ to process
        np.savetxt("plots/data/histogram/MIMOLLRfreq_+1_" + mod.mod_name + "_{}x{}_{}".format(sim_params.Nt, sim_params.Nr, sim_params.L) + fn_tikz + ".dat",
                   np.c_[llr_list0, llr2_list0],
                   )
        np.savetxt("plots/data/histogram/MIMOLLRfreq_-1_" + mod.mod_name + "_{}x{}_{}".format(sim_params.Nt, sim_params.Nr, sim_params.L) + fn_tikz + ".dat",
                   np.c_[llr_list1, llr2_list1],
                   )
        print('Saved MIMO LLR histograms to "plots/".')

    print('Simulation ended.')
# EOF
