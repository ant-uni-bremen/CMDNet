#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:05:19 2019

@author: beck
"""

# LOADED PACKAGES
import os
import sys
import time

# Python packages
import numpy as np
import tensorflow as tf
import yaml
# Tensorflow/Keras packages
# from tensorflow import keras
from tensorflow.keras import backend as KB
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Reshape
# For online training DNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, Nadam

# Own packages
import my_modules.mycom as com
import my_modules.myequ as equ
import my_modules.myfunc as mf
import my_modules.mymathops as mop
import my_modules.mymimoch as mch
import my_modules.mytraining as mt

# Include parent folder
sys.path.append('..')  # Include parent folder, where own packages lie
# Include current folder, where start simulation script and packages lie
sys.path.append('.')


# Functions exclusive to this file

class CMD_graph():
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


def CMD_initpar(M, L, typ, k=0):
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
        train_hist2 = mt.training_history()
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


class DetNet_graph():
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
        path = os.path.join('models', train_par.mod.mod_name,
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


class MMNet_graph():
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
        path = os.path.join('models', train_par.mod.mod_name,
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


def MIMODNN(sim_par, Nh, NL=1, soft=0):
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


def opt_sel(opt_name='sgd', lrs=None):
    '''Selection of an Keras optimizer
    INPUT
    opt_name: Optimizer as string
    lrs: Learning rate schedule (optional)
    OUTPUT
    opt: Optimizer object from Keras
    '''
    if lrs == None:
        if opt_name.casefold() == 'adam':
            # NOTE: learning_rate = 10**-4 (HyperCMDfull)
            opt = Adam()
        elif opt_name.casefold() == 'sgd':
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
            # lr = 1e-6/2, momentum = 0.9, nesterov = True
            opt = SGD(learning_rate=lrs, momentum=0.9, nesterov=True)
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


class Data_gen_mimo2():
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


if __name__ == '__main__':
    #     my_func_main()
    # def my_func_main():

    # Set working directory to script directory
    # Path of script being executed
    path0 = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path0)

    # Load settings
    # sim_settings.yaml, sim_settings_llr.yaml, sim_settings_online.yaml, sim_settings_code.yaml
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
        mt.tf_enable_GPU(0, 8)          # disable GPU
        # mt.tf_enable_GPU(1, 1)        # enable GPU / default

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
        detnet = DetNet_graph(train_params, sim_set['fn_ext'])

    if llr_mode == 2:
        # LLR comparison
        # Load DetNet [fixed comparison to this version of DetNet]
        # if sim_set['algo'] == 'DetNet':
        train_ebn0_detnet = [4, 11]
        fn_ext_detnet = '_defsnr'
        train_params2 = mf.simulation_parameters(
            sim_params.Nt, sim_params.Nr, sim_params.L, sim_params.mod, train_set['batch_size'], train_ebn0_detnet, sim_params.rho)
        detnet_llr = DetNet_graph(train_params2, fn_ext_detnet)

    # Load MMNet or OAMPNet
    if sim_set['algo'] in ['MMNet', 'OAMPNet']:
        mmnet = MMNet_graph(train_params, sim_set['algo'], sim_set['fn_ext'])

    # CMDNet/CMD
    if sim_set['algo'] == 'CMD_fixed':
        # Choose parameters / starting point
        if load_set['train'] == 1:
            fn2 = mf.filename_module(
                'trainhist_', 'curves', 'CMD', '_binary_splin', sim_set)
            train_hist = mt.training_history()
            train_hist.filename = fn2.pathfile
            train_hist.dict2obj(saveobj.load(train_hist.filename))
            # [delta0, taui0], _ = train_hist.sel_best_weights()
            [delta0, taui0] = train_hist.params[-1]
        else:
            [delta0, taui0] = CMD_initpar(
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
        opt = opt_sel(opt_name=train_set['opt'], lrs=lrs)

    if sim_set['algo'] in cmd_algos:
        # Choose parameters / starting point
        if sim_set['algo'] == 'CMDpar':
            [delta0, taui0] = CMD_initpar(
                sim_params.mod.M, sim_params.L, train_set['start_point'], k=3)
        else:
            [delta0, taui0] = CMD_initpar(
                sim_params.mod.M, sim_params.L, train_set['start_point'])

        # Initialization
        train_hist = mt.training_history()
        train_hist.filename = fn.pathfile  # Set path for checkpoint saves
        # Load results if already trained
        if load_set['train'] == 1:
            train_hist.dict2obj(saveobj.load(train_hist.filename))

        # Create graph
        if sim_set['algo'] == 'HyperCMD':
            CMDgraph = mt.HyperCMD_graph(train_params, delta0, taui0, soft=train_set['soft'],
                                         multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        elif sim_set['algo'] == 'CMDpar':
            CMDgraph = mt.CMDpar_graph(train_params, delta0, taui0, soft=train_set['soft'],
                                       multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        else:
            CMDgraph = CMD_graph(train_params, delta0, taui0, soft=train_set['soft'],
                                 multiloss=train_set['multiloss'], binopt=sim_set['binopt'])
        # Train
        train_hist = train(train_set['Nepoch'], train_hist, CMDgraph.train_inputs, CMDgraph.loss,
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
                dnnmimo, loss = MIMODNN(
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
                        opt = opt_sel(opt_name=train_set['opt'], lrs=lrs)
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
                        # data_gen_mimo = Data_gen_mimo2()    # Easy debugging, just comment out and watch data_gen_mimo.it
                        history = dnnmimo.fit(data_gen_mimo(data_gen),
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
