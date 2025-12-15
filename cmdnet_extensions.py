#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 12:25:20 2024

@author: beck
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as KB


# -------- CMDNet Extensions (May be included in final PhD Thesis) ---------------


class HyperCMDGraph():
    '''HyperCMD graph object
    CMDNet with parameters computed by a hypernetwork for one-shot learning with/conditioned on each channel matrix and noise variance
    '''
    # Class Attribute
    name = 'Tensorflow HyperCMD graph object'
    # Initializer / Instance Attributes

    def __init__(self, train_par, delta0, taui0, soft=0, multiloss=0, binopt=0):
        self.inputs = 0
        self.train_inputs = 0
        self.outputs = 0
        self.loss = 0
        self.params = 0
        self.hinputs = 0
        self.houtputs = 0
        if train_par.mod.M == 2 and binopt == 1:
            self.create_graph_bin(train_par, delta0, taui0,
                                  soft=soft, multiloss=multiloss)
        else:
            self.create_graph(train_par, delta0, taui0,
                              soft=soft, multiloss=multiloss)
        # Calculate Output (Soft information / symbols) given Input
        self.predict = KB.function(self.inputs, self.outputs)
        self.hypernetwork = KB.function(self.hinputs, self.houtputs)
        # self.cross_entr = KB.function(self.train_inputs, self.loss) # Loss output
    # Instance methods

    def create_hypernetwork_basic(self, sigmat, delta0, taui0):
        '''Hypernetwork basic architecture: One NN per parameter for function approximation
        INPUT: Conditioned variables
        sigmat: Noise tensor
        delta0/taui0: Starting point of hyperparameters
        OUTPUT
        taui, delta: Hyperparameters
        '''
        # Hypernetwork parameters
        Np = 6
        W1_taui = KB.variable(
            value=0.01 * np.ones((taui0.shape[0], Np)))  # default: 0.01
        W2_taui = KB.variable(value=0.01 * np.ones((taui0.shape[0], Np)))
        W1_delta = KB.variable(value=0.01 * np.ones((delta0.shape[0], Np)))
        W2_delta = KB.variable(value=0.01 * np.ones((delta0.shape[0], Np)))
        B1_taui = KB.variable(value=np.zeros((taui0.shape[0], Np)))
        B1_delta = KB.variable(value=np.zeros((delta0.shape[0], Np)))
        b2_taui = KB.variable(value=taui0)
        b2_delta = KB.variable(value=delta0)
        # 8 params
        self.params.extend([b2_delta, b2_taui, W1_taui,
                           W2_taui, W1_delta, W2_delta, B1_taui, B1_delta])
        # Hypernetwork function
        taui = KB.abs(KB.sum(W2_taui * KB.relu(W1_taui *
                      KB.expand_dims(sigmat, axis=-1) + B1_taui), axis=-1) + b2_taui)
        # KB.abs() in original implementation -> only positive delta...
        delta = KB.sum(W2_delta * KB.relu(W1_delta *
                       KB.expand_dims(sigmat, axis=-1) + B1_delta), axis=-1) + b2_delta
        self.hinputs = [sigmat]
        self.houtputs = [taui, delta]
        return taui, delta

    def create_hypernetwork_sigma2(self, sigmat, delta0, taui0):
        '''Hypernetwork sigma2 -> fully connected per parameter set
        INPUT: conditioned variables
        sigmat: noise tensor
        delta0/taui0: starting point of hyperparameters
        OUTPUT
        taui, delta: hyperparameters
        '''
        init = tf.contrib.layers.xavier_initializer(
            dtype=tf.float32)  # tf.float16, tf.float32, tf.float64
        # Hypernetwork parameters
        N_L1 = 6
        N_L2 = 75
        w1_taui = KB.variable(value=init((N_L1, )))  # default: 0.01
        W2_taui = KB.variable(value=init((N_L2, N_L1)))
        W3_taui = KB.variable(value=init((taui0.shape[0], N_L2)))
        w1_delta = KB.variable(value=init((N_L1, )))
        W2_delta = KB.variable(value=init((N_L2, N_L1)))
        W3_delta = KB.variable(value=init((delta0.shape[0], N_L2)))
        b1_taui = KB.variable(value=np.zeros((N_L1)))
        b1_delta = KB.variable(value=np.zeros((N_L1)))
        b2_taui = KB.variable(value=np.zeros((N_L2)))
        b2_delta = KB.variable(value=np.zeros((N_L2)))
        b3_taui = KB.variable(value=taui0)
        b3_delta = KB.variable(value=delta0)
        # 12 params
        self.params.extend([b3_delta, b3_taui, w1_taui, W2_taui, W3_taui,
                           w1_delta, W2_delta, W3_delta, b1_taui, b1_delta, b2_taui, b2_delta])
        # Hypernetwork function
        taui = KB.abs(KB.batch_dot(tf.expand_dims(W3_taui, axis=0), KB.relu(KB.batch_dot(
            tf.expand_dims(W2_taui, axis=0), KB.relu(w1_taui * sigmat + b1_taui)) + b2_taui)) + b3_taui)
        delta = KB.batch_dot(tf.expand_dims(W3_delta, axis=0), KB.relu(KB.batch_dot(tf.expand_dims(
            W2_delta, axis=0), KB.relu(w1_delta * sigmat + b1_delta)) + b2_delta)) + b3_delta
        self.hinputs = [sigmat]
        self.houtputs = [taui, delta]
        return taui, delta

    def create_hypernetwork_sigma(self, sigmat, delta0, taui0):
        '''Hypernetwork sigma -> fully connected between all parameters
        INPUT: Conditioned variables
        sigmat: Noise tensor
        delta0/taui0: Starting point of hyperparameters
        OUTPUT
        taui, delta: Hyperparameters
        '''
        init = tf.contrib.layers.xavier_initializer(
            dtype=tf.float32)  # tf.float16, tf.float32, tf.float64
        # Hypernetwork parameters
        N_L1 = 6
        N_L2 = 75
        w1 = KB.variable(value=init((N_L1, )))
        W2 = KB.variable(value=init((N_L2, N_L1)))
        W3 = KB.variable(value=init((taui0.shape[0] + delta0.shape[0], N_L2)))
        b1 = KB.variable(value=np.zeros((N_L1)))
        b2 = KB.variable(value=np.zeros((N_L2)))
        b3 = KB.variable(value=np.concatenate((taui0, delta0), axis=0))
        # b3 = KB.variable(value = np.zeros((taui0.shape[0] + delta0.shape[0])))
        # 6 params
        self.params.extend([b3, w1, W2, W3, b1, b2])
        # Hypernetwork function
        hparams = KB.batch_dot(tf.expand_dims(W3, axis=0), KB.elu(KB.batch_dot(
            tf.expand_dims(W2, axis=0), KB.elu(w1 * sigmat + b1)) + b2)) + b3
        taui = KB.abs(hparams[:, :taui0.shape[0]])  # * taui0
        delta = hparams[:, taui0.shape[0]:]  # * delta0
        self.hinputs = [sigmat]
        self.houtputs = [taui, delta]
        return taui, delta

    def create_hypernetwork_full(self, sigmat0, Ht, delta0, taui0, typus=0, qr=0):
        '''Hypernetwork fully connected -> too many parameters
        INPUT: Conditioned variables
        sigmat: Noise tensor
        Ht: Channel tensor
        delta0/taui0: Starting point of hyperparameters
        OUTPUT
        taui, delta: Hyperparameters
        '''
        init = tf.contrib.layers.xavier_initializer(
            dtype=tf.float32)  # tf.float16, tf.float32, tf.float64
        sigmat = KB.expand_dims(sigmat0, axis=-1)
        if qr == 0:
            # 1. Full matrix input
            Hvec = tf.reshape(Ht, shape=[-1, Ht.shape[1] * Ht.shape[2]])
        else:
            # 2. QR decomposition input to hypernetwork, extract upper triangle
            r, c = np.triu_indices(Ht.get_shape().as_list()[-1])
            Hvec = tf.transpose(tf.gather_nd(tf.transpose(Ht, [1, 2, 0]), np.concatenate(
                (r[:, np.newaxis], c[:, np.newaxis]), axis=-1).tolist()), [1, 0])
        hinput = tf.concat([sigmat, Hvec], axis=-1)  # noise std or variance?
        # Hypernetwork parameters
        N_L1 = hinput.get_shape().as_list()[-1]
        if typus == 0:
            # 1. Scalar
            Nt = 1
            N_L2 = 75
            N_L3 = taui0.shape[0] + delta0.shape[0]
        else:
            # 2. Vector
            Nt = Ht.get_shape().as_list()[-1]
            N_L3 = (taui0.shape[0] + delta0.shape[0]) * Nt
            N_L2 = int((N_L3 + N_L1) / 2)
        W1 = KB.variable(value=init((N_L1, N_L1)))
        W2 = KB.variable(value=init((N_L2, N_L1)))
        # W3 = KB.variable(value = init((N_L2, N_L2)))
        # W4 = KB.variable(value = init((N_L2, N_L2)))
        WN = KB.variable(value=init((N_L3, N_L2)))
        b1 = KB.variable(value=np.zeros((N_L1)))
        b2 = KB.variable(value=np.zeros((N_L2)))
        # b3 = KB.variable(value = np.zeros((N_L2)))
        # b4 = KB.variable(value = np.zeros((N_L2)))
        # bN = KB.variable(value = np.concatenate((taui0, delta0), axis = 0))
        bN = KB.variable(value=np.zeros((N_L3)))
        # self.params.extend([bN, W1, W2, WN, b1, b2])
        # 6 params
        self.params.extend([bN, W1, W2, WN, b1, b2])
        # Hypernetwork function
        l1 = KB.elu(KB.batch_dot(tf.expand_dims(W1, axis=0), hinput) + b1)
        l2 = KB.elu(KB.batch_dot(tf.expand_dims(W2, axis=0), l1) + b2)
        # l3 = KB.elu(KB.batch_dot(tf.expand_dims(W3, axis = 0), l2) + b3)
        # l4 = KB.elu(KB.batch_dot(tf.expand_dims(W4, axis = 0), l3) + b4)
        hparams = KB.batch_dot(tf.expand_dims(WN, axis=0), l2) + bN
        if typus == 0:
            # 1. Scalar
            taui = KB.abs(hparams[:, :taui0.shape[0]])  # * taui0
            delta = hparams[:, taui0.shape[0]:]  # * delta0
        else:
            # 2. Vector
            hparams_vec = tf.reshape(
                hparams, shape=[-1, Nt, taui0.shape[0] + delta0.shape[0]])
            taui = KB.abs(hparams_vec[:, :, :taui0.shape[0]])  # * taui0
            delta = hparams_vec[:, :, taui0.shape[0]:]  # * delta0
        self.hinputs = [sigmat0, Ht]
        self.houtputs = [taui, delta]
        return taui, delta

    def create_hypernetwork(self, sigmat0, Ht, delta0, taui0):
        '''Hypernetwork convolutional -> reduce number of parameters with convolutional and max-pooling layers
        INPUT: Conditioned variables
        sigmat: Noise tensor
        Ht: Channel tensor
        delta0/taui0: Starting point of hyperparameters
        OUTPUT
        taui, delta: Hyperparameters
        '''
        def conv_net(data, weights, biases):
            # Define 2D convolutional function
            def conv2d(x, W, b, strides=1):
                x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
                x = tf.nn.bias_add(x, b)
                return tf.nn.relu(x)
            # Convolution layers
            conv1 = conv2d(data[0], weights['c1'],
                           biases['c1'])  # [28, 28, 16]
            conv2 = conv2d(conv1, weights['c2'], biases['c2'])  # [28, 28, 16]
            pool1 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding='SAME')
            # [14,14,16]

            conv3 = conv2d(pool1, weights['c3'], biases['c3'])  # [14, 14, 32]
            conv4 = conv2d(conv3, weights['c4'], biases['c4'])  # [14, 14, 32]
            pool2 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[
                                   1, 2, 2, 1], padding='SAME')
            # [7,7,32]

            # Flatten
            flat = tf.reshape(
                pool2, [-1, np.prod(pool2.get_shape().as_list()[1:])])
            flat2 = tf.concat([data[1], flat], axis=-1)
            # [7*7*32] = [1568]

            # Fully connected layer
            fc1 = tf.add(
                tf.matmul(flat2, weights['d1']), biases['d1'])  # [128]
            fc1 = tf.nn.relu(fc1)  # [128]

            # Dropout
            # if training:
            #     fc1 = tf.nn.dropout(fc1, rate=0.2)

            # Output
            out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])  # [10]
            return out

        init = tf.contrib.layers.xavier_initializer(
            dtype=tf.float32)  # tf.float16, tf.float32, tf.float64
        N_output = taui0.shape[0] + delta0.shape[0]
        # 12 params
        weights = {
            # Convolution Layers
            'c1': KB.variable(name='W1', value=init((3, 3, 1, 16))),
            'c2': KB.variable(name='W2', value=init((3, 3, 16, 16))),
            'c3': KB.variable(name='W3', value=init((3, 3, 16, 32))),
            'c4': KB.variable(name='W4', value=init((3, 3, 32, 32))),

            # Dense Layers
            # adjust to NrxNt
            'd1': KB.variable(name='W5', value=init((16*16*32+1, 128))),
            'out': KB.variable(name='W6', value=init((128, N_output))),
        }
        biases = {
            # Convolution Layers
            'c1': KB.variable(name='B1', value=np.zeros((16))),
            'c2': KB.variable(name='B2', value=np.zeros((16))),
            'c3': KB.variable(name='B3', value=np.zeros((32))),
            'c4': KB.variable(name='B4', value=np.zeros((32))),

            # Dense Layers
            'd1': KB.variable(name='B5', value=np.zeros((128))),
            'out': KB.variable(name='B6', value=np.zeros((N_output))),
        }

        sigmat = KB.expand_dims(sigmat0, axis=-1)
        for key in weights:
            self.params.append(weights[key])
        for key in biases:
            self.params.append(biases[key])
        # self.params.extend([bN, W1, W2, W3, W4, WN, b1, b2, b3, b4])
        # Hypernetwork function
        Ht2 = tf.expand_dims(Ht, axis=-1)
        hparams = conv_net([Ht2, sigmat], weights, biases)
        taui = KB.abs(hparams[:, :taui0.shape[0]])  # * taui0
        delta = hparams[:, taui0.shape[0]:]  # * delta0
        self.hinputs = [sigmat0, Ht]
        self.houtputs = [taui, delta]
        return taui, delta

    def create_graph_bin(self, train_par, delta0, taui0, soft=0, multiloss=0):
        '''Binary concrete MAP Detection Unfolding
        Create graph with variables and placeholders
        soft: MSE: = 1 | Binary_Cross_Entropy: = 0
        multiloss:  Multiloss on: = 1
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
        self.params = []

        # Loss
        loss = []

        # Preprocessing
        alphat = KB.constant(value=alpha)
        sigmat = KB.expand_dims(sigmat0, axis=-1)
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)

        # Hypernetwork
        typus = 1
        qr = 0
        taui, delta = self.create_hypernetwork_full(
            sigmat0, Ht, delta0, taui0, typus=typus, qr=qr)

        # UNFOLDING / Starting point of first layer
        s = KB.transpose(KB.expand_dims(s0)) * KB.ones_like(Ht[:, 0, :])
        if typus == 0:
            # 1. Scalar
            taui_abs = KB.expand_dims(KB.abs(taui[:, 0]), axis=-1)
        else:
            # 2. Vector
            taui_abs = KB.abs(taui[:, :, 0])
        xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)

        for iteration in range(0, it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = 1 / 2 * taui_abs * (1 - xt ** 2)
            grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
            # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
            # Gradient/ResNet Layer
            if typus == 0:
                # 1. Scalar
                s = s - KB.expand_dims(delta[:, iteration], axis=-1) * grad_L
            else:
                # 2. Vector
                s = s - delta[:, :, iteration] * grad_L

            # Start of new iteration
            if typus == 0:
                # 1. Scalar
                # no negative values for tau !
                taui_abs = KB.expand_dims(
                    KB.abs(taui[:, iteration + 1]), axis=-1)
            else:
                # 2. Vector
                taui_abs = KB.abs(taui[:, :, iteration + 1])
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
            # output layer and objective function
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
        soft: MSE: = 1 | Binary_Cross_Entropy: = 0
        multiloss:  Multiloss on: = 1
        '''
        ce = 0  # Exact or truncated softmax implementation?
        [Nr, Nt, it, mod, alpha] = [train_par.Nr, train_par.Nt,
                                    train_par.L, train_par.mod, train_par.mod.alpha]
        N_class = mod.M

        # CMD Unfolding
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
        self.params = []
        # Loss
        loss = []

        # Preprocessing
        alphat = KB.constant(value=alpha)
        m = KB.expand_dims(KB.expand_dims(
            KB.variable(value=mod.m), axis=0), axis=0)
        sigmat = KB.expand_dims(KB.expand_dims(sigmat0, axis=-1), axis=-1)
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)

        # Hypernetwork
        taui, delta = self.create_hypernetwork_full(sigmat0, Ht, delta0, taui0)

        # UNFOLDING / Starting point of first layer
        G = KB.expand_dims(G0, axis=0) * \
            KB.expand_dims(KB.ones_like(Ht[:, 0, :]), axis=-1)
        taui_abs = KB.expand_dims(KB.abs(taui[0]), axis=-1)
        ft = KB.softmax((KB.log(alphat) + G) * KB.abs(taui_abs), axis=-1)
        xt = KB.sum(ft * m, axis=-1)

        for iteration in range(0, it):
            xHH = KB.batch_dot(xt, HH)
            grad_x = taui_abs * (ft * m - ft * KB.expand_dims(xt, axis=-1))
            grad_L = sigmat ** 2 * (1 - KB.exp(-G)) + \
                grad_x * KB.expand_dims(xHH - yH, axis=-1)
            # grad_L =  (1 - KB.exp(-G)) + 1 / sigmat ** 2 * grad_x * KB.expand_dims(xHH - yH, axis = -1) # original version
            # Gradient/ResNet Layer
            G = G - KB.expand_dims(delta[iteration], axis=-1) * grad_L
            # Start of new iteration
            # no negative values for tau !
            taui_abs = KB.expand_dims(KB.abs(taui[:, iteration + 1]), axis=-1)
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
                        lloss = (iteration + 1) * KB.mean(
                            KB.mean(KB.categorical_crossentropy(out_true, -(-ft), axis=-1), axis=-1))
                    else:
                        lloss = (iteration + 1) * KB.mean(
                            KB.mean(KB.categorical_crossentropy(out_true, ft, axis=-1), axis=-1))
                loss.append(lloss)

        if multiloss == 1:
            loss = KB.sum(loss)
        else:
            # output layer and objective function
            if soft == 1:
                # 2. mean should be a sum for overall MSE scaling with Nt
                loss = KB.mean(KB.mean((out_true - xt) ** 2, axis=-1))
            else:
                # 2. mean should be a sum since q factorizes
                if ce == 0:
                    # -(-ft) for truncated cross entropy with faster training convergence
                    loss = KB.mean(KB.mean(KB.categorical_crossentropy(
                        out_true, -(-ft), axis=-1), axis=-1))
                else:
                    loss = KB.mean(KB.mean(KB.categorical_crossentropy(
                        out_true, ft, axis=-1), axis=-1))
        self.loss = loss
        self.outputs = [ft, xt]
        return self.train_inputs, self.outputs, self.loss, self.params


class CMDparGraph():
    '''CMDpar graph object
    Idea: Introduce parallel branches of CMD local gradient descent optimization with multiple
    starting points
    '''
    # Class Attribute
    name = 'Tensorflow CMDpar graph object'
    # Initializer / Instance Attributes

    def __init__(self, train_par, delta0, taui0, soft=0, multiloss=0, binopt=0):
        self.inputs = 0
        self.train_inputs = 0
        self.outputs = 0
        self.loss = 0
        self.params = 0
        if train_par.mod.M == 2 and binopt == 1:
            self.create_graph_bin(train_par, delta0, taui0,
                                  soft=soft, multiloss=multiloss)
        else:
            print('No multi-class version implemented!')
            # self.create_graph(train_par, delta0, taui0, soft = soft, multiloss = multiloss)
        # Calculate output (Soft information/symbols) given Input
        self.predict = KB.function(self.inputs, self.outputs)
        # self.cross_entr = KB.function(self.train_inputs, self.loss) # Loss output
    # Instance methods

    def create_graph_bin(self, train_par, delta0, taui0, soft=0, multiloss=0):
        '''Binary concrete MAP Detection Unfolding
        Create graph with variables and placeholders
        soft: MSE: = 1 | Binary_Cross_Entropy: = 0
        multiloss:  Multiloss on: = 1
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
        s00 = np.concatenate((np.zeros((Nt, 1)), np.log(
            3) * np.ones((Nt, 1)), - np.log(3) * np.ones((Nt, 1))), axis=-1)
        # s00 = np.concatenate((np.zeros((Nt, 1)), 3 * np.ones((Nt, 1)), - 3 * np.ones((Nt, 1))), axis = -1)
        s0 = KB.variable(value=s00[:, 0])
        s0_2 = KB.variable(value=s00[:, 1:])
        taui = KB.variable(value=taui0)
        delta = KB.variable(value=delta0)
        self.params = [delta, taui, s0_2]
        # Loss
        # loss = []

        # Preprocessing
        alphat = KB.constant(value=alpha)
        sigmat = KB.expand_dims(sigmat0, axis=-1)
        HH = KB.batch_dot(KB.permute_dimensions(Ht, (0, 2, 1)), Ht)
        yH = KB.batch_dot(yt, Ht)

        # UNFOLDING / Starting point of first layer
        xt_L = []
        ft_L = []
        ml_crit = []
        for itpar in range(0, s00.shape[-1]):
            # Go through parallel branches
            if itpar == 0:
                s = KB.transpose(KB.expand_dims(s0)) * \
                    KB.ones_like(Ht[:, 0, :])
            else:
                s = KB.transpose(KB.expand_dims(
                    s0_2[:, itpar - 1])) * KB.ones_like(Ht[:, 0, :])

            taui_abs = KB.abs(taui[0, itpar])
            # * taui_abs) # not necessary
            xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2)

            # xt0 = [np.sum(train_par.mod.alpha * train_par.mod.m, axis = -1), np.ones((Nt, )), -np.ones((Nt, ))]
            # xt = KB.variable(value = xt0[itpar]) * KB.ones_like(Ht[:, 0, :])

            for iteration in range(0, it):
                xHH = KB.batch_dot(xt, HH)
                grad_x = 1 / 2 * taui_abs * (1 - xt ** 2)
                grad_L = sigmat ** 2 * KB.tanh(s / 2) + grad_x * (xHH - yH)
                # grad_L = KB.tanh(s / 2) + 1 / sigmat ** 2 * grad_x * (xHH - yH) # original version
                # Gradient/ResNet Layer
                s = s - delta[iteration, itpar] * grad_L

                # Start of new iteration
                # no negative values for tau !
                taui_abs = KB.abs(taui[iteration + 1, itpar])
                xt = KB.tanh((KB.log(1 / alphat - 1) + s) / 2 * taui_abs)
                xt2 = KB.expand_dims(xt, axis=-1)
                if (train_par.mod.m == np.array([-1, 1])).all():
                    # [q(x = -1), q(x = 1)]
                    ft = KB.concatenate(
                        [(1 - xt2) / 2, (1 + xt2) / 2], axis=-1)
                else:
                    # [q(x = 1), q(x = -1)]
                    ft = KB.concatenate(
                        [(1 + xt2) / 2, (1 - xt2) / 2], axis=-1)
            L = KB.mean((yt - KB.batch_dot(xt, Ht)) ** 2, axis=-1) / 2 + sigmat[:, 0] ** 2 * (
                KB.sum(s, axis=-1) + 2 * KB.sum(KB.log(1 + KB.exp(-s)), axis=-1))
            ml_crit.append(KB.expand_dims(L, axis=-1))
            xt_L.append(KB.expand_dims(xt, axis=-1))
            ft_L.append(KB.expand_dims(ft, axis=-1))

        # Choose best branch
        min_ind = tf.argmin(KB.concatenate(ml_crit, axis=-1),
                            axis=-1, output_type=tf.int32)
        xt_L = KB.concatenate(xt_L, axis=-1)
        ft_L = KB.concatenate(ft_L, axis=-1)
        num_examples = tf.cast(tf.shape(yt)[0], dtype=min_ind.dtype)
        idx = tf.stack([tf.range(num_examples), min_ind], axis=-1)
        xt = tf.gather_nd(tf.transpose(xt_L, [0, 2, 1]), idx)
        ft = tf.gather_nd(tf.transpose(ft_L, [0, 3, 1, 2]), idx)
        # xt = tf.case([(tf.equal(min_ind, KB.zeros_like(min_ind)), lambda: xt_L[0]), (tf.equal(min_ind, KB.ones_like(min_ind)), lambda: xt_L[1])], default = lambda: xt_L[2])
        # ft = tf.switch_case(min_ind, branch_fns = {0: lambda: ft_L[0], 1: lambda: ft_L[1]}, default = lambda: ft_L[2])

        if multiloss == 1:
            print('No multiloss support.')

        # output layer and objective function
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
