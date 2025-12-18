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
import time

# Python packages
import numpy as np
import tensorflow as tf
# Tensorflow/Keras packages
# from tensorflow import keras
from tensorflow.keras import backend as KB
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Reshape
# For online training DNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Nadam

# Own packages
import utilities.my_functions as mf
import utilities.my_training_tf1 as mt

# Functions exclusive to this file


class CMDGraph():
    '''CMDNet graph object
    Original implementation from:
    Beck, E.; Bockelmann, C.; Dekorsy, A. CMDNet: Learning a Probabilistic Relaxation of
    Discrete Variables for Soft Detection with Low Complexity. IEEE Trans. Commun. 2021, 69, 8214-8227
    '''
    # Class Attribute
    name = 'Tensorflow CMDNet graph object'
    # Initializer / Instance Attributes

    def __init__(self, train_par, delta0, taui0, soft=0, multiloss=0, binopt=0):
        '''CMDNet Initialization
        binopt: Version of CMDNet for two-classes/binary decision with BPSK or QPSK (CMDNet_bin)
        '''
        self.inputs = 0
        self.train_inputs = 0
        self.outputs = 0
        self.loss = 0
        self.params = 0
        if train_par.mod.M == 2 and binopt == 1:
            self.create_graph_bin(train_par, delta0, taui0,
                                  soft=soft, multiloss=multiloss)
        else:
            self.create_graph(train_par, delta0, taui0,
                              soft=soft, multiloss=multiloss)
        # Calculate Output (Soft information/symbols) given Input
        self.predict = KB.function(self.inputs, self.outputs)
        # self.cross_entr = KB.function(self.train_inputs, self.loss)   # Loss output
    # Instance methods

    def create_graph_bin(self, train_par, delta0, taui0, soft=0, multiloss=0):
        '''Binary concrete MAP Detection Unfolding - CMDNet bin
        Create graph with variables and placeholders
        train_par: Training parameters object
        delta0: Step size starting point
        taui0: Inverse softmax temperature starting point
        soft: MSE: = 1 | Binary_Cross_Entropy: = 0
        multiloss: Multiloss on: = 1
        '''
        [Nr, Nt, it] = [train_par.Nr, train_par.Nt, train_par.L]
        # Select alpha according to paper notation m = [-1, 1]
        if (train_par.mod.m == np.array([-1, 1])).all():
            alpha = train_par.mod.alpha[:, 0]
        else:
            alpha = train_par.mod.alpha[:, 1]

        # CMD Unfolding
        # Create graph with variables and placeholders
        yt = KB.placeholder(name="yt", shape=(None, Nr))
        Ht = KB.placeholder(name="Ht", shape=(None, Nr, Nt))
        sigmat0 = KB.placeholder(name="sigmat", shape=(None, ))
        if soft == 1:
            out_true = KB.placeholder(name="out_true", shape=(None, Nt))
        else:
            out_true = KB.placeholder(
                name="out_true", shape=(None, Nt, 2))  # two classes

        # Inputs
        self.train_inputs = [yt, Ht, sigmat0, out_true]
        self.inputs = self.train_inputs[0:-1]
        # Variables
        s0 = KB.variable(value=np.zeros((Nt)))
        taui = KB.variable(value=taui0)
        delta = KB.variable(value=delta0)
        self.params = [delta, taui]
        # Loss
        loss = []

        # Preprocessing
        alphat = KB.constant(value=alpha)
        sigmat = KB.expand_dims(sigmat0, axis=-1)
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)

        # UNFOLDING / Starting point of first layer
        s = KB.transpose(KB.expand_dims(s0)) * KB.ones_like(Ht[:, 0, :])
        taui_abs = KB.abs(taui[0])
        xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)

        for iteration in range(0, it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = 1 / 2 * taui_abs * (1 - xt ** 2)
            grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            s = s - delta[iteration] * grad_L

            # Start of new gradient descent iteration
            # no negative values for tau !
            taui_abs = KB.abs(taui[iteration + 1])
            xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
            xt2 = KB.expand_dims(xt, axis=-1)
            if (train_par.mod.m == np.array([-1, 1])).all():
                # [q(x = -1), q(x = 1)]
                ft = KB.concatenate([(1 - xt2) / 2, (1 + xt2) / 2], axis=-1)
            else:
                # [q(x = 1), q(x = -1)]
                ft = KB.concatenate([(1 + xt2) / 2, (1 - xt2) / 2], axis=-1)
            if multiloss == 1:
                if soft == 1:
                    # 2. mean should be a sum for overall MSE scaling with Nt
                    lloss = (iteration + 1) * \
                        KB.mean(KB.mean((out_true - xt) ** 2, axis=-1))
                else:
                    # 2. mean should be a sum since q factorizes
                    lloss = (iteration + 1) * KB.mean(
                        KB.mean(KB.categorical_crossentropy(out_true, ft, axis=-1), axis=-1))
                loss.append(lloss)

        if multiloss == 1:
            loss = KB.sum(loss)
        else:
            # Output layer and objective function
            if soft == 1:
                # 2. mean should be a sum for overall MSE scaling with Nt
                loss = KB.mean(KB.mean((out_true - xt) ** 2, axis=-1))
            else:
                # 2. mean should be a sum since q factorizes
                loss = KB.mean(KB.mean(KB.categorical_crossentropy(
                    out_true, ft, axis=-1), axis=-1))
        self.loss = loss
        self.outputs = [ft, xt]
        return self.train_inputs, self.outputs, self.loss, self.params

    def create_graph(self, train_par, delta0, taui0, soft=0, multiloss=0):
        '''Concrete MAP Detection Unfolding - CMDNet
        Create graph with variables and placeholders
        train_par: Training parameters object
        delta0: Step size starting point
        taui0: Inverse softmax temperature starting point
        soft: MSE: = 1 | Binary_Cross_Entropy: = 0
        multiloss:  Multiloss on: = 1
        '''
        ce = 0  # Truncated or exact softmax implementation? 0/1
        [Nr, Nt, it, mod, alpha] = [train_par.Nr, train_par.Nt,
                                    train_par.L, train_par.mod, train_par.mod.alpha]
        N_class = mod.M

        # CMD Unfolding - CMDNet
        # Create graph with variables and placeholders
        yt = KB.placeholder(name="yt", shape=(None, Nr))
        Ht = KB.placeholder(name="Ht", shape=(None, Nr, Nt))
        sigmat0 = KB.placeholder(name="sigmat", shape=(None, ))
        if soft == 1:
            out_true = KB.placeholder(name="out_true", shape=(None, Nt))
        else:
            out_true = KB.placeholder(
                name="out_true", shape=(None, Nt, N_class))

        # Inputs
        self.train_inputs = [yt, Ht, sigmat0, out_true]
        self.inputs = self.train_inputs[0:-1]
        # Variables
        G0 = KB.variable(value=np.zeros((Nt, N_class)))
        taui = KB.variable(value=taui0)
        delta = KB.variable(value=delta0)
        self.params = [delta, taui]
        # Loss
        loss = []

        # Preprocessing
        alphat = KB.constant(value=alpha)
        m = KB.expand_dims(KB.expand_dims(
            KB.variable(value=mod.m), axis=0), axis=0)
        sigmat = KB.expand_dims(KB.expand_dims(sigmat0, axis=-1), axis=-1)
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)

        # UNFOLDING / Starting point of first layer
        G = KB.expand_dims(G0, axis=0) * \
            KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis=-1)
        taui_abs = KB.abs(taui[0])
        ft = KB.softmax((KB.log(alphat) + G) * taui_abs, axis=-1)
        xt = KB.sum(ft * m, axis=-1)

        for iteration in range(0, it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = taui_abs * (ft * m - ft * KB.expand_dims(xt, axis=-1))
            # Implemented version from article
            grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + \
                grad_x * KB.expand_dims(xHH - yH, axis=-1)
            # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1)   # Original version from article
            # Gradient/ResNet Layer
            G = G - delta[iteration] * grad_L
            # Start of new gradient descent iteration
            # -> No negative values for tau !
            taui_abs = KB.abs(taui[iteration + 1])
            logits = (KB.log(alphat) + G) * taui_abs
            ft = KB.softmax((KB.log(alphat) + G) * taui_abs, axis=-1)
            xt = KB.sum(ft * m, axis=-1)
            if multiloss == 1:
                if soft == 1:
                    # 2. mean should be a sum for overall MSE scaling with Nt
                    lloss = (iteration + 1) * \
                        KB.mean(KB.mean((out_true - xt) ** 2, axis=-1))
                else:
                    # 2. mean should be a sum since q factorizes
                    if ce == 0:
                        # -(-ft) for truncated cross entropy with faster training convergence
                        lloss = (iteration + 1) * KB.mean(KB.mean(KB.categorical_crossentropy(
                            out_true, ft, axis=-1, from_logits=False), axis=-1))
                    else:
                        lloss = (iteration + 1) * KB.mean(KB.mean(KB.categorical_crossentropy(
                            out_true, logits, axis=-1, from_logits=True), axis=-1))
                loss.append(lloss)

        if multiloss == 1:
            loss = KB.sum(loss)
        else:
            # Output layer and objective function
            if soft == 1:
                # 2. mean should be a sum for overall MSE scaling with Nt
                loss = KB.mean(KB.mean((out_true - xt) ** 2, axis=-1))
            else:
                # 2. mean should be a sum since q factorizes
                if ce == 0:
                    # -(-ft) for truncated cross entropy with faster training convergence
                    loss = KB.mean(KB.mean(KB.categorical_crossentropy(
                        out_true, ft, axis=-1, from_logits=False), axis=-1))
                else:
                    loss = KB.mean(KB.mean(KB.categorical_crossentropy(
                        out_true, logits, axis=-1, from_logits=True), axis=-1))

        self.loss = loss
        self.outputs = [ft, xt]

        return self.train_inputs, self.outputs, self.loss, self.params


def train(Nepoch, train_hist, inputs, loss, params, opt, data_gen, saveobj, it_checkp=100, sv_checkp=0, sel_best_weights=0, esteps=0, lr_dyn=0):
    '''Training in TensorFlow 1 with Keras Backend
    INPUT
    Nepoch: Number of training iterations
    train_hist: Object with history of training and validation loss, etc.
    inputs: Inputs of graph
    loss: Loss of optimization problem
    params: Parameter in graph to be trained
    opt: Optmizer (keras)
    data_gen: Data generator / data_gen.data_val: includes Validation data
    saveobj: Object with saving functionality
    add_params: Additional parameters that are fixed, but to be stored
    OPTIONAL
    it_checkp: Iterations until validation checkpoint
    sv_checkp: If model is saved at checkpoint
    sel_best_weights: If best weights are selected in the end
    esteps: Iterations w/o improvement until early stopping
    lr_dyn: Learning rate dynamic according to function lr_schedule()
    OUTPUT
    train_hist: Object with history of training and validation loss, etc.
    '''
    updates = opt.get_updates(params=params, loss=loss)
    train = KB.function(inputs, [loss], updates=updates)
    test = KB.function(inputs, [loss])
    # History object
    if train_hist.epoch == []:
        # First evaluation, if empty
        start_time = time.time()
        val_loss = test(data_gen.data_val)[0]
        train_hist(0, 0, val_loss, params, opt,
                   time.time() - start_time)  # no train_loss
        train_hist.printh()
    else:
        # Initialization from train_history -> TODO: Move higher ? -> e.g., self.sel_epoch = 5 -> erase content afterwards in all lists???
        # idea epoch_reset: define function in train_hist resetting after sel_epoch
        # KB.batch_set_value(list(zip(params, train_hist.params[-1])))
        train_hist.sel_weights(params, -1)
        opt.from_config(train_hist.opt_config[-1])
        opt.set_weights(train_hist.opt_params[-1])
    if Nepoch != 0:
        start_time = time.time()
        for epoch in range(train_hist.epoch[-1], Nepoch):
            # Create new training data set every batch -> only 1 epoch
            data_gen()
            train_loss = train(data_gen.data)[0]
            if np.isnan(train_loss):
                print('Encountered NaN in training loss: Stopped training.')
                break
            # Validation after it_checkp steps / Last evaluation is saved for sure
            if (epoch + 1) % it_checkp == 0 or epoch + 1 == Nepoch:
                # val_loss = test(data_val)[0]
                val_loss = test(data_gen.data_val)[0]
                train_hist(epoch + 1, train_loss, val_loss, params, opt, time.time() -
                           start_time, add_params=data_gen.data_opt)  # tracking loop time w/o overhead
                if sv_checkp == 1 or epoch + 1 >= Nepoch:  # Save if checkpoint or last iteration
                    saveobj.save(train_hist.filename, train_hist.obj2dict())
                train_hist.printh()
                # Early Stopping / Decreasing learning rate schedule
                if esteps > 0 and train_hist.early_stopping(esteps):
                    if lr_dyn == 1:
                        # learning rate schedule, e.g., here exponential
                        func_lr = lr_schedule(
                            train_hist.opt_config[-1]['learning_rate'])
                        KB.set_value(opt.lr, func_lr)
                        print('Decreased learning rate.')
                    else:
                        print('Early stopping...')
                        # save if early stopping
                        saveobj.save(train_hist.filename,
                                     train_hist.obj2dict())
                        break
                start_time = time.time()

    # Select best weights
    if sel_best_weights == 1:
        train_hist.set_best_weights(params)
    return train_hist


def lr_schedule(lr):
    '''Computes learning rate schedule
    lr: learning rate
    lr_new: New learning rate after update step
    '''
    lr_new = lr / 2
    return lr_new


def cmd_initpar(M, L, typ, k=0):
    '''Calculate a good heuristic starting point for CMD / loading starting point
    M: Number of classes / Modulation order
    L: Number of layers / iterations
    'typ': linear/constant
    k: Number of starting points, e.g., for CMDpar
    delta0: Starting point of iterative step sizes
    taui0: Starting point of iterative softmax temperatures
    '''
    # Finetuning of starting point has to be done at this point
    # Maximum parameter values
    if M >= 4:
        # NOTE: * 2 shows better results for 4ASK or 16QAM
        tau_max = 1 / (M - 1) * 2
    else:
        # Default softmax temperature that makes the concrete distribution convex instead of concave
        # and thus gives better approximation of discrete random variables
        tau_max = 1 / (M - 1)
    delta_max = 1
    # Minimum parameter values
    tau_min = 0.1                   # * tau_max # default: 0.1
    delta_min = 0.1                 # * delta_max # default: 0.1
    if typ.casefold() == 'linear':
        # Linear decrease
        tau0 = tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max - (delta_max - delta_min) / L * np.linspace(0, L, L)
        taui0 = 1 / tau0
    elif typ.casefold() == 'const':
        # Constant
        tau0 = tau_max * np.ones(L + 1)
        delta0 = delta_max * np.ones(L)
        taui0 = 1 / tau0
    elif typ.casefold() == 'load':
        # Load starting point
        # Choose it here in the code
        # MIMO 64x64 (default)
        sim_set = {
            'Mod': 'QPSK',
            'Nr': 64,
            'Nt': 64,
            'L': 64,
        }
        # default: _binary_tau0.1
        fn2 = mf.filename_module(
            'trainhist_', 'curves', 'CMD', '_binary_tau0.1', sim_set)
        # MIMO 16x16
        # sim_set = {
        #     'Mod': 'QPSK',
        #     'Nr': 16,
        #     'Nt': 16,
        #     'L': 16,
        # }
        # fn2 = mf.filename_module('trainhist_', 'curves', 'CMD', '_binary', sim_set) # default: _binary
        ##
        saveobj2 = mf.savemodule('npz')
        train_hist2 = mt.TrainingHistory()
        train_hist2.dict2obj(saveobj2.load(fn2.pathfile))
        [delta0, taui0] = train_hist2.params[-1]
    else:
        # Default: Only linear decrease in taui, delta constant
        tau0 = tau_max - (tau_max - tau_min) / L * np.linspace(0, L, L + 1)
        delta0 = delta_max * np.ones(L)
        taui0 = 1 / tau0

    if k != 0:
        # For more than one starting point, e.g., CMDpar
        tau0 = tau0[:, np.newaxis].repeat((k), axis=-1)
        delta0 = delta0[:, np.newaxis].repeat((k), axis=-1)
        taui0 = 1 / tau0

    return delta0, taui0


class DetNetGraph():
    '''DetNet graph object
    For loading a DetNet graph from original source code and evaluating it inside this script
    '''
    # Class Attribute
    name = 'Tensorflow DetNet graph object'
    # Initializer / Instance Attributes

    def __init__(self, train_par, fn_ext):
        self.inputs = 0
        self.outputs = 0
        self.graph = 0
        self.create_graph(train_par, fn_ext)
        with self.graph.as_default():
            session_one = tf.Session()
            with session_one.as_default():
                # Output soft information/symbols
                self.predict = KB.function(self.inputs, self.outputs)
    # Instance methods

    def generate_filename(self, train_par, fn_ext):
        '''Generate filename for DetNet load
        train_par: Training parameter object
        fn_ext: Filename extension
        '''
        filename = 'DetNet_' + train_par.mod.mod_name + '_{}_{}_{}_snr{}_{}'.format(
            train_par.Nt, train_par.Nr, train_par.L, train_par.EbN0_range[0], train_par.EbN0_range[1]) + fn_ext
        path = os.path.join('LearningToDetect_for_CMDNet', 'models_cmdnet', train_par.mod.mod_name,
                            '{}x{}'.format(train_par.Nt, train_par.Nr))
        path2 = os.path.join(path, filename)
        pathfile = os.path.join(path2, filename)
        return pathfile, path2

    def create_graph(self, train_par, fn_ext):
        '''Load and create DetNet graph from file
        train_par: Training parameter object
        fn_ext: Filename extension
        '''
        # Reload the model/graph into the session
        pathfile, path = self.generate_filename(train_par, fn_ext)
        new_saver = tf.train.import_meta_graph(pathfile + '.meta')
        # Workaround: Use global session...
        new_saver.restore(KB.get_session(), tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()

        # Collect inputs and outputs
        # Inputs
        HY = graph.get_tensor_by_name("HY:0")
        HH = graph.get_tensor_by_name("HH:0")
        # X = graph.get_tensor_by_name("X:0")
        # X_IND = graph.get_tensor_by_name("X_IND:0")
        self.inputs = [HY, HH]
        # Outputs
        S1 = graph.get_tensor_by_name("S1_" + str(train_par.L - 1) + ":0")
        # Reshape for QAM16
        if train_par.mod.M == 4:
            # # Reshape soft output acc. to script notation
            # S2 = tf.transpose(S2, perm = [0, 3, 1, 2])
            S3 = graph.get_tensor_by_name("S3_" + str(train_par.L - 2) + ":0")
            S3 = S3 / tf.expand_dims(tf.reduce_sum(S3, axis=-1), axis=-1)
            S3 = tf.expand_dims(S3, axis=-1)
            S4_re = tf.concat([S3[:, :, 0] + S3[:, :, 1] + S3[:, :, 2] + S3[:, :, 3],
                               S3[:, :, 4] + S3[:, :, 5] +
                               S3[:, :, 6] + S3[:, :, 7],
                               S3[:, :, 8] + S3[:, :, 9] +
                               S3[:, :, 10] + S3[:, :, 11],
                               S3[:, :, 12] + S3[:, :, 13] + S3[:, :, 14] + S3[:, :, 15]], axis=-1)
            S4_re = S4_re / \
                tf.expand_dims(tf.reduce_sum(S4_re, axis=-1), axis=-1)
            S4_im = tf.concat([S3[:, :, 0] + S3[:, :, 4] + S3[:, :, 8] + S3[:, :, 12],
                               S3[:, :, 1] + S3[:, :, 5] +
                               S3[:, :, 9] + S3[:, :, 13],
                               S3[:, :, 2] + S3[:, :, 6] +
                               S3[:, :, 10] + S3[:, :, 14],
                               S3[:, :, 3] + S3[:, :, 7] + S3[:, :, 11] + S3[:, :, 15]], axis=-1)
            S4_im = S4_im / \
                tf.expand_dims(tf.reduce_sum(S4_im, axis=-1), axis=-1)
            S4 = tf.concat([S4_re, S4_im], axis=1)
            # # switch output order from -3,-1,1,3 to [-3,-1,3,1]
            perm = [0, 1, 3, 2]
            out = tf.gather(S4, perm, axis=-1)
        else:
            # Own modifications to output to ensure valid pmf
            S2 = graph.get_tensor_by_name("Out_" + str(train_par.L - 1) + ":0")
            S2_clip = tf.clip_by_value(S2, 0, 1)
            S2t = tf.reshape(S2_clip, (-1, train_par.Nt, train_par.mod.M))
            # out = S2t
            out = S2t / tf.expand_dims(tf.reduce_sum(S2t, axis=-1), axis=-1)
        # SER = graph.get_tensor_by_name("SER:0")
        self.outputs = [out, S1]
        # Original output (BPSK)
        # last_layer = np.array(sess.run([S2], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND:BATCH_X_IND}))
        # last_layer = np.clip(last_layer, -1, 1)
        # ind1_last_layer = last_layer[:, 0:2*sim_params.Nt:2]
        # ind2_last_layer = last_layer[:, 1:2*sim_params.Nt:2]
        # fr = ind1_last_layer
        self.graph = graph
        return self.inputs, self.outputs, self.graph


class MMNetGraph():
    '''MMNet graph object
    For loading a MMNet graph from original source code and evaluating it inside this script
    '''
    # Class Attribute
    name = 'Tensorflow MMNet graph object'
    # Initializer / Instance Attributes

    def __init__(self, train_par, algo, fn_ext):
        self.inputs = 0
        self.outputs = 0
        self.create_graph(train_par, algo, fn_ext)
        # Calculate Output (Soft information/symbols) given Input
        self.predict = KB.function(self.inputs, self.outputs)
    # Instance methods

    def generate_filename(self, train_par, algo, fn_ext):
        '''Generate filename for MMNet load
        algo: Algorithm name; see MMNet implementation
        train_par: Training parameter object
        fn_ext: Filename extension
        '''
        filename = algo + '_' + train_par.mod.mod_name + '_{}_{}_{}_snr{}_{}'.format(
            train_par.Nt, train_par.Nr, train_par.L, train_par.EbN0_range[0], train_par.EbN0_range[1]) + fn_ext
        path = os.path.join('MMNet_for_CMDNet', 'learning_based', 'models_cmdnet', train_par.mod.mod_name,
                            '{}x{}'.format(train_par.Nt, train_par.Nr))
        path2 = os.path.join(path, filename)
        pathfile = os.path.join(path2, filename)
        return pathfile, path2

    def create_graph(self, train_par, algo, fn_ext):
        '''Load and create MMNet graph from file
        algo: Algorithm name; see MMNet implementation
        train_par: Training parameter object
        fn_ext: Filename extension
        OUTPUT
        inputs = [snr_db_max, snr_db_min, batch_size, H]
        outputs = [fr, z, x_out, x, constellation]
        TODO: Noise variance tensor is missing for soft output (LLR) calculation (use with channel coding)
        '''
        pathfile, path = self.generate_filename(train_par, algo, fn_ext)
        new_saver = tf.train.import_meta_graph(pathfile + '.meta')
        # Workaround: Use global session...
        new_saver.restore(KB.get_session(), tf.train.latest_checkpoint(path))
        graph = tf.get_default_graph()

        # Collect inputs and outputs
        # Inputs
        H = graph.get_tensor_by_name("H:0")
        snr_db_max = graph.get_tensor_by_name("snr_db_max:0")
        snr_db_min = graph.get_tensor_by_name("snr_db_min:0")
        batch_size = graph.get_tensor_by_name("batch_size:0")
        # lr = graph.get_tensor_by_name("lr:0")
        # train_flag = graph.get_tensor_by_name("train_flag:0")
        # , batch_size, lr, train_flag]
        self.inputs = [snr_db_max, snr_db_min, batch_size, H]
        # Outputs
        # sqrt(2) for consistency with implementation
        x_NN = graph.get_tensor_by_name("Out:0") * tf.math.sqrt(2.0)
        x_out = x_NN[-1, :, :]
        # sqrt(2) for consistency with implementation
        x = graph.get_tensor_by_name("input:0") * tf.math.sqrt(2.0)
        # sqrt(2) for consistency with implementation
        constellation = graph.get_tensor_by_name("const:0") * tf.math.sqrt(2.0)
        indices = graph.get_tensor_by_name("indices:0")
        z = tf.one_hot(indices, depth=train_par.mod.M, dtype=tf.int32)
        # Reshape for QAM16
        if train_par.mod.M == 4:
            constellation = tf.gather(constellation, [0, 1, 3, 2])
            z = tf.gather(z, [0, 1, 3, 2], axis=-1)
        elif train_par.mod.M == 16:
            print('Not yet implemented!!!')
        # Return class in addition to symbols
        clr = tf.argmin(tf.abs(tf.expand_dims(x_out, axis=-1) - tf.expand_dims(
            tf.expand_dims(constellation, axis=0), axis=0)) ** 2, axis=-1)
        fr = tf.one_hot(clr, depth=train_par.mod.M, dtype=tf.int32)
        self.outputs = [fr, z, x_out, x, constellation]
        # Numpy equivalent
        # z = keras.utils.to_categorical(indices, num_classes = sim_params.mod.M).astype('int32')
        # # Compute output
        # clr = np.argmin(np.abs(s[:, :, np.newaxis] - constellation[np.newaxis, np.newaxis, :]) ** 2, axis = -1)
        # fr = keras.utils.to_categorical(clr, num_classes = sim_params.mod.M).astype('int32')
        return self.inputs, self.outputs


def mimo_dnn(sim_par, Nh, NL=1, soft=0):
    '''Simple DNN Detector/Classifier with residual connections for MIMO Equalization
    INPUT
    sim_par: simulation parameter object
    Nh: Intermediate layer width
    NL: Number of indermediate layers
    soft: hard output with probability estimate (0: default CE loss) / soft output with estimate of x (1: MSE loss)
    (Nrx: Number of receive antennas)
    (M: Number of classes/constellation symbols)
    (Ntx: Number of transmit antennas)
    OUTPUT
        dnnmimo: DNN model
    '''
    Nrx = sim_par.Nr
    M = sim_par.mod.M
    Ntx = sim_par.Nt
    inputs = Input(shape=(Nrx, ))
    layer = inputs

    # Intermediate layers
    for indL in range(0, NL):
        layer_prev = layer
        if indL == 0 and Nh != Nrx:
            # If input and layer dimension do not match, introduce a linear layer to the first residual connection
            layer_prev = Dense(Nh, activation='linear')(layer_prev)
        layer = Dense(Nh, activation='relu',
                      kernel_initializer='he_uniform')(layer_prev)
        # Residual connection improves training speed/convergence and expressive power
        layer = Add()([layer_prev, layer])

    if soft == 1:
        # Final linear layer for estimation
        outputs = Dense(Ntx, activation='linear')(layer)
        loss = 'mean_squared_error'
    else:
        # Final softmax layer for classification
        softmax = []
        for _ in range(0, Ntx):
            layer_last = Dense(M, activation='softmax')(layer)
            layer_last2 = Reshape((1, M))(layer_last)
            softmax.append(layer_last2)
        outputs = Concatenate(axis=1)(softmax)
        loss = 'categorical_crossentropy'

    dnnmimo = Model(inputs=inputs, outputs=outputs)
    return dnnmimo, loss


def optimizer_select(opt_name='sgd', lrs=None):
    '''Selection of an Keras optimizer
    INPUT
    opt_name: Optimizer as string
    lrs: Learning rate schedule (optional)
    OUTPUT
    opt: Optimizer object from Keras
    '''
    if lrs is None:
        if opt_name.casefold() == 'adam':
            # NOTE: learning_rate = 10**-4 (HyperCMDfull)
            opt = Adam()
        elif opt_name.casefold() == 'sgd':
            opt = SGD()
        elif opt_name.casefold() == 'sgd_nesterov':
            # lr = 1e-6/2, momentum = 0.9, nesterov = True
            opt = SGD(momentum=0.9, nesterov=True)
        elif opt_name.casefold() == 'nadam':
            opt = Nadam()
        else:
            print('Optimizer not available!')
    else:
        if opt_name.casefold() == 'adam':
            # NOTE: learning_rate = 10**-4 (HyperCMDfull)
            opt = Adam(learning_rate=lrs)
        elif opt_name.casefold() == 'sgd':
            opt = SGD(learning_rate=lrs)
        elif opt_name.casefold() == 'sgd_nesterov':
            # lr = 1e-6/2, momentum = 0.9, nesterov = True
            opt = SGD(learning_rate=lrs, momentum=0.9,
                      nesterov=True)  # clipvalue=100
        elif opt_name.casefold() == 'nadam':
            opt = Nadam(learning_rate=lrs)
        else:
            print('Optimizer not available!')
    return opt


def data_gen_mimo(data_generator):
    '''Data generator for memory-efficient DNN online learning
    with huge datasets; comes at the expense of slower training
    INPUT
    data_generator: Model data generator
    OUTPUT
    data[0], data[-1]: Received signal and chosen output class or symbol as tuple
    '''
    while True:
        data = data_generator.gen_online_mimodata()
        yield (data[0], data[-1])


class DataGenMIMO2():
    '''Data generator data_gen_mimo as object
    Used for debugging as it counts the number of execution iterations
    '''
    name = 'Data generator object'

    def __init__(self):
        self.it = 0

    def __call__(self, data_generator):
        while True:
            data = data_generator.gen_online_mimodata()
            self.it = self.it + 1
            yield (data[0], data[-1])


# EOF
