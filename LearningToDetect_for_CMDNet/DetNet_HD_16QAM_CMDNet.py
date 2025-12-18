#!/usr/bin/env python

"""
This file is used to train and test the DetNet architecture in the hard decision output scenario.
The constellation used is 16QAM and the channel is complex
all parameters were optimized and trained over the 15X25 iid channel, changing the channel might require parameter tuning

Notice that the run time analysis presented in the paper was made on a numpy version of the tensorflow network.
writen by Neev Samuel based on the paper:
    "Learning to detect, Neev Samuel,Tzvi Diskin,Ami Wiesel"

contact by neev.samuel@gmail.com

"""
import os
import tensorflow as tf
import numpy as np
import time as tm
import math
import sys
import pickle as pkl

# Own additions
import scipy as sp
import scipy.integrate as integrate
import utilities.my_mimo_channel as mymimo

# start here

# own parameter
no_gpu = 0                  # if without gpu
compl = 1                   # 1
rho = 0                     # 0, 10, 20
cell_sector = 120           # 120
it_print = 100              # 100
fn_ext = '_HD_test'         # '_HD'
save_dir = ''
# Save in CMDNet root folder
# save_dir = os.path.join(os.path.dirname(__file__), '..')
# save_dir = os.path.abspath(save_dir)
if no_gpu == 1:
    num_GPU = 0
    num_cores = 8
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': 1, 'GPU': num_GPU})
    sess = tf.InteractiveSession(config=config)
else:
    sess = tf.InteractiveSession()

# parameters
"""
Parameters
K - size of x
N - size of y
snrdb_low - the lower bound of noise db used during training
snr_high - the higher bound of noise db used during training
L - number of layers in DetNet
v_size = size of auxiliary variable at each layer
hl_size - size of hidden layer at each DetNet layer (the dimention the layers input are increased to
startingLearningRate1 - the initial step size of the gradient descent algorithm when train phase without noise
startingLearningRate2 - the initial step size of the gradient descent algorithm when train phase with noise
decay_factor & decay_step_size - each decay_step_size steps the learning rate decay by decay_factor
train_iter - number of train iterations
train_iter_no_noise - number of train iterations without noise
train_batch_size - batch size during training phase
test_iter - number of test iterations
test_batch_size  - batch size during testing phase
LOG_LOSS - equal 1 if loss of each layer should be sumed in proportion to the layer depth, otherwise all losses have the same weight 
res_alpha- the proportion of the previuos layer output to be added to the current layers output (view ResNet article)
snrdb_low_test & snrdb_high_test & num_snr - when testing, num_snr different SNR values will be tested, uniformly spread between snrdb_low_test and snrdb_high_test 
symbols - the possible symbols the consellation has (when converted to the real setting as discribed in the paper)
"""
K2 = 64
N2 = 64
K = int(K2 / 2)                         # default: 15, Nt * 2, HD: 15
N = int(N2 / 2)                         # default: 25, Nr * 2, HD: 30
ebn0db_low = 1                          # default: 1 vs. 10 (CMD)
ebn0db_high = 8                         # default: 8 vs. 33 (CMD)
snr_shift = 10 * np.log10(2 * np.log2(4))
snrdb_low = ebn0db_low + snr_shift      # default: 7
snrdb_high = ebn0db_high + snr_shift    # default: 14
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)
L = K2
v_size = 4*(2*K)
hl_size = 8*(2*K)
startingLearningRate1 = 0.0003
startingLearningRate2 = 0.0003
decay_factor1 = 0.97
decay_factor2 = 0.97
decay_step_size1 = 1000
decay_step_size2 = 1000
train_iter = 200000
train_iter_no_noise = 5000
n0 = 0.5

train_batch_size = 3000
test_iter = 100
test_batch_size = 2000
LOG_LOSS = 1
res_alpha = 0.9
num_snr = 6
snrdb_low_test = 8.0
snrdb_high_test = 13.0
symbols = np.array([-3, -1, 1, 3])

# Original parameters
# K = 15
# N = 25
# snrdb_low = 7.0
# snrdb_high = 14.0
# snr_low = 10.0 ** (snrdb_low/10.0)
# snr_high = 10.0 ** (snrdb_high/10.0)
# L=30
# v_size = 4*(2*K)
# hl_size = 8*(2*K)
# startingLearningRate1 = 0.0003
# startingLearningRate2 = 0.0003
# decay_factor1 = 0.97
# decay_factor2 = 0.97
# decay_step_size1 = 1000
# decay_step_size2 = 1000
# train_iter = 200000
# train_iter_no_noise = 5000
# n0 = 0.5

# train_batch_size = 3000
# test_iter= 100
# test_batch_size = 2000
# LOG_LOSS = 1
# res_alpha=0.9
# num_snr = 6
# snrdb_low_test=8.0
# snrdb_high_test=13.0
# symbols = np.array([-3,-1,1,3])


print('16QAM hard decision DetNet parameters')
print(K)
print(N)
print(snrdb_low)
print(snrdb_high)
print(snr_low)
print(snr_high)
print(L)
print(v_size)
print(hl_size)
print(startingLearningRate1)
print(startingLearningRate2)
print(decay_factor1)
print(decay_factor2)
print(decay_step_size1)
print(decay_step_size2)
print(train_iter)
print(train_batch_size)
print(test_iter)
print(test_batch_size)
print(res_alpha)
print(num_snr)
print(snrdb_low_test)
print(snrdb_high_test)

"""Data generation for train and test phases
In this example, both functions are the same.
This duplication is in order to easily allow testing cases where the test is over different distributions of data than in the training phase.
e.g. training over gaussian i.i.d. channels and testing over a specific constant channel.
currently both test and train are over i.i.d gaussian channel.
"""


def generate_channel(Nb, Nr, Nt, compl=0, rho=0, Phi_R12=0):
    '''Generates complex or real-valued [Nr]x[Nt] channel according to one ring correlation model
    Nb: Number of channel realizations
    Nr: Number of receive antennas
    Nt: Number of transmit antennas
    compl: Complex (1) / real (0)
    Phi_R12: correlation matrices
    '''
    if rho > 1:
        Hr = mymimo.generate_channel_onering(Nb, Nr, Nt, Phi_R12, compl=compl)
    else:
        Hr = mymimo.generate_channel(Nb, Nr, Nt, compl=compl, rho=rho)
    return Hr


# One Ring correlation matrices
if rho > 1:
    if compl == 1:
        Phi_R12 = mymimo.mimo_OneRingModel(N=int(N / 2), angularSpread=rho, cell_sector=cell_sector,compl=compl)
    else:
        Phi_R12 = mymimo.mimo_OneRingModel(N=N, angularSpread=rho, cell_sector=cell_sector,compl=compl)
else:
    Phi_R12 = 0


def generate_data_iid_test(B, K, N, snr_low, snr_high, WithNoise, compl=1, rho=0, Phi_R12=0):
    x_R = np.random.randint(4, size=(B, K))
    x_R = x_R * 2
    x_R = x_R - 3

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3

    x_ind = np.zeros([B, K, 16])
    for i in range(B):
        for ii in range(K):
            if x_R[i, ii] == -3 and x_I[i, ii] == -3:
                x_ind[i, ii, 0] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == -1:
                x_ind[i, ii, 1] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == 1:
                x_ind[i, ii, 2] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == 3:
                x_ind[i, ii, 3] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == -3:
                x_ind[i, ii, 4] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == -1:
                x_ind[i, ii, 5] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == 1:
                x_ind[i, ii, 6] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == 3:
                x_ind[i, ii, 7] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == -3:
                x_ind[i, ii, 8] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == -1:
                x_ind[i, ii, 9] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == 1:
                x_ind[i, ii, 10] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == 3:
                x_ind[i, ii, 11] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == -3:
                x_ind[i, ii, 12] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == -1:
                x_ind[i, ii, 13] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == 1:
                x_ind[i, ii, 14] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == 3:
                x_ind[i, ii, 15] = 1

    x_ = np.concatenate((x_R, x_I), axis=1)

    # H_R = np.random.randn(B, N, K)
    # H_I = np.random.randn(B, N, K)
    # H_ = np.zeros([B, 2 * N, 2 * K])
    H_ = generate_channel(B, 2 * N, 2 * K, compl, rho,
                          Phi_R12)  # * np.sqrt(2 * N)
    H_R = H_[:, :N, :K]
    H_I = H_[:, N:, :K]

    y_ = np.zeros([B, 2 * N])

    # because of non-normalized modulation given in symbols=[-3,-1,1,3] (!!!)
    a = np.sqrt(3 / (4 ** 2 - 1))
    w_R = np.random.randn(B, N) / a
    w_I = np.random.randn(B, N) / a
    w = np.concatenate((w_R, w_I), axis=1)

    Hy_ = np.zeros([B, 2 * K])  # x_ * 0 # -> leads to quantized Hy...
    HH_ = np.zeros([B, 2 * K, 2 * K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        # H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
        #                    np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        # H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :]) + WithNoise*w[i, :] * \
            np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_, H_R, H_I, x_R, x_I, w_R, w_I, x_ind


def generate_data_train(B, K, N, snr_low, snr_high, WithNoise, compl=1, rho=0, Phi_R12=0):
    x_R = np.random.randint(4, size=(B, K))
    x_R = x_R * 2
    x_R = x_R - 3

    x_I = np.random.randint(4, size=(B, K))
    x_I = x_I * 2
    x_I = x_I - 3

    x_ind = np.zeros([B, K, 16])
    for i in range(B):
        for ii in range(K):
            if x_R[i, ii] == -3 and x_I[i, ii] == -3:
                x_ind[i, ii, 0] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == -1:
                x_ind[i, ii, 1] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == 1:
                x_ind[i, ii, 2] = 1
            if x_R[i, ii] == -3 and x_I[i, ii] == 3:
                x_ind[i, ii, 3] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == -3:
                x_ind[i, ii, 4] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == -1:
                x_ind[i, ii, 5] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == 1:
                x_ind[i, ii, 6] = 1
            if x_R[i, ii] == -1 and x_I[i, ii] == 3:
                x_ind[i, ii, 7] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == -3:
                x_ind[i, ii, 8] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == -1:
                x_ind[i, ii, 9] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == 1:
                x_ind[i, ii, 10] = 1
            if x_R[i, ii] == 1 and x_I[i, ii] == 3:
                x_ind[i, ii, 11] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == -3:
                x_ind[i, ii, 12] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == -1:
                x_ind[i, ii, 13] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == 1:
                x_ind[i, ii, 14] = 1
            if x_R[i, ii] == 3 and x_I[i, ii] == 3:
                x_ind[i, ii, 15] = 1

    x_ = np.concatenate((x_R, x_I), axis=1)

    # H_R = np.random.randn(B, N, K)
    # H_I = np.random.randn(B, N, K)
    # H_ = np.zeros([B, 2 * N, 2 * K])
    H_ = generate_channel(B, 2 * N, 2 * K, compl, rho,
                          Phi_R12)  # * np.sqrt(2 * N)
    H_R = H_[:, :N, :K]
    H_I = H_[:, N:, :K]

    y_ = np.zeros([B, 2 * N])

    # because of non-normalized modulation given in symbols=[-3,-1,1,3] (!!!)
    a = np.sqrt(3 / (4 ** 2 - 1))
    w_R = np.random.randn(B, N) / a
    w_I = np.random.randn(B, N) / a
    w = np.concatenate((w_R, w_I), axis=1)

    Hy_ = x_ * 0
    HH_ = np.zeros([B, 2 * K, 2 * K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        # H = np.concatenate((np.concatenate((H_R[i, :, :], -1 * H_I[i, :, :]), axis=1),
        #                     np.concatenate((H_I[i, :, :], H_R[i, :, :]), axis=1)), axis=0)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace() / (2 * K)
        # H_[i, :, :] = H
        y_[i, :] = H.dot(x_[i, :]) + WithNoise*w[i, :] * \
            np.sqrt(tmp_snr)/np.sqrt(SNR)
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_, H_R, H_I, x_R, x_I, w_R, w_I, x_ind


def find_nearest(values):
    values = values + 3
    values = values/2
    values = tf.clip_by_value(values, 0, 3)
    values = tf.round(values)
    values = values * 2
    values = values - 3
    return values


def piecewise_linear_soft_sign(x):
    t = tf.constant(0.1)
    y = -3+tf.nn.relu(x+2+t)/(tf.abs(t)+0.00001)-tf.nn.relu(x+2-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x+t)/(tf.abs(t)+0.00001) - \
        tf.nn.relu(x-t)/(tf.abs(t)+0.00001)+tf.nn.relu(x-2+t) / \
        (tf.abs(t)+0.00001)-tf.nn.relu(x-2-t)/(tf.abs(t)+0.00001)
    return y


def affine_layer(x, input_size, output_size, Layer_num):
    W = tf.Variable(tf.random_normal([input_size, output_size], stddev=0.01))
    w = tf.Variable(tf.random_normal([1, output_size], stddev=0.01))
    y = tf.matmul(x, W)+w
    return y


def relu_layer(x, input_size, output_size, Layer_num):
    y = tf.nn.relu(affine_layer(x, input_size, output_size, Layer_num))
    return y


def sign_layer(x, input_size, output_size, Layer_num):
    y = affine_layer(x, input_size, output_size, Layer_num)
    return y


# tensorflow placeholders, the input given to the model in order to train and test the network
HY = tf.placeholder(tf.float32, shape=[None, 2*K])
X = tf.placeholder(tf.float32, shape=[None, 2*K])
HH = tf.placeholder(tf.float32, shape=[None, 2*K, 2*K])
X_IND = tf.placeholder(tf.float32, shape=[None, K, 16])


batch_size = tf.shape(HY)[0]


S1 = []
S1.append(tf.zeros([batch_size, 2*K], name="S1"))
S2 = []
S2.append(tf.zeros([batch_size, 16*K], name="S2"))
V = []
V.append(tf.zeros([batch_size, v_size]))
LOSS = []
LOSS.append(tf.zeros([]))
BER = []
BER.append(tf.zeros([]))
delta = tf.Variable(tf.zeros(L*2, 1))

# The architecture of DetNet
for i in range(1, L):
    temp1 = tf.matmul(tf.expand_dims(S1[-1], 1), HH)
    temp1 = tf.squeeze(temp1, 1)
    Z1 = S1[-1] - delta[(i-1) * 2]*HY + delta[(i-1) * 2 + 1]*temp1
    Z = tf.concat([Z1, V[-1]], 1)
    ZZ = relu_layer(Z, (2*K) + v_size, hl_size, 'relu'+str(i))
    S2.append(sign_layer(ZZ, hl_size, K*16, 'sign'+str(i)))
    S2[i] = tf.identity((1-res_alpha)*S2[i]+res_alpha*S2[i-1], name="S2")

    V.append(affine_layer(ZZ, hl_size, v_size, 'aff'+str(i)))
    V[i] = (1-res_alpha)*V[i]+res_alpha*V[i-1]

    S3 = tf.reshape(S2[i], [batch_size, K, 16], name="S3")

    temp_0 = S3[:, :, 0]
    temp_1 = S3[:, :, 1]
    temp_2 = S3[:, :, 2]
    temp_3 = S3[:, :, 3]
    temp_4 = S3[:, :, 4]
    temp_5 = S3[:, :, 5]
    temp_6 = S3[:, :, 6]
    temp_7 = S3[:, :, 7]
    temp_8 = S3[:, :, 8]
    temp_9 = S3[:, :, 9]
    temp_10 = S3[:, :, 10]
    temp_11 = S3[:, :, 11]
    temp_12 = S3[:, :, 12]
    temp_13 = S3[:, :, 13]
    temp_14 = S3[:, :, 14]
    temp_15 = S3[:, :, 15]

    S1_real = -3.0*temp_0 +\
              -3.0*temp_1 +\
              -3.0*temp_2 +\
              -3.0*temp_3 +\
              -1.0*temp_4 +\
              -1.0*temp_5 +\
              -1.0*temp_6 +\
              -1.0*temp_7 +\
        1.0*temp_8 +\
        1.0*temp_9 +\
        1.0*temp_10 +\
        1.0*temp_11 +\
        3.0*temp_12 +\
        3.0*temp_13 +\
        3.0*temp_14 +\
        3.0*temp_15

    S1_im = -3.0*temp_0 +\
        -1.0*temp_1 +\
        1.0*temp_2 +\
        3.0*temp_3 +\
        -3.0*temp_4 +\
        -1.0*temp_5 +\
        1.0*temp_6 +\
        3.0*temp_7 +\
        -3.0*temp_8 +\
        -1.0*temp_9 +\
        1.0*temp_10 +\
        3.0*temp_11 +\
        -3.0*temp_12 +\
        -1.0*temp_13 +\
        1.0*temp_14 +\
        3.0*temp_15

    S1.append(tf.concat([S1_real, S1_im], 1, name="S1"))

    X_IND_reshaped = tf.reshape(X_IND, [batch_size, 16*K])
    if LOG_LOSS == 1:
        # /tf.reduce_mean(tf.square(X - X_LS),1)))
        LOSS.append(
            np.log(i)*tf.reduce_mean(tf.reduce_mean(tf.square(X_IND_reshaped - S2[-1]), 1)))
    else:
        # /tf.reduce_mean(tf.square(X - X_LS),1)))
        LOSS.append(tf.reduce_mean(tf.reduce_mean(
            tf.square(X_IND_reshaped - S2[-1]), 1)))
    BER.append(tf.reduce_mean(
        tf.cast(tf.not_equal(X_IND, tf.round(S3)), tf.float32)))
Max_Val = tf.reduce_max(S3, axis=2, keep_dims=True)
Greater = tf.greater_equal(S3, Max_Val)
BER2 = tf.round(tf.cast(Greater, tf.float32))
BER3 = tf.not_equal(BER2, X_IND)
BER4 = tf.reduce_sum(tf.cast(BER3, tf.float32), 2)
BER5 = tf.cast(tf.greater(BER4, 0), tf.float32)
SER = tf.reduce_mean(BER5)

TOTAL_LOSS = tf.add_n(LOSS)

saver = tf.train.Saver()

global_step1 = tf.Variable(0, trainable=False)
learning_rate1 = tf.train.exponential_decay(
    startingLearningRate1, global_step1, decay_step_size1, decay_factor1, staircase=True)
train_step1 = tf.train.AdamOptimizer(learning_rate1).minimize(TOTAL_LOSS)

global_step2 = tf.Variable(0, trainable=False)
learning_rate2 = tf.train.exponential_decay(
    startingLearningRate2, global_step2, decay_step_size2, decay_factor2, staircase=True)
train_step2 = tf.train.AdamOptimizer(learning_rate2).minimize(TOTAL_LOSS)

init_op = tf.initialize_all_variables()

sess.run(init_op)


# Save model into file
def check_path(pathfile, verbose=0):
    '''Check for existing path and file, respectively
    '''
    path = os.path.dirname(pathfile)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        if verbose == 1:
            print('Created new directory.')
    else:
        if os.path.isfile(pathfile):
            os.remove(pathfile)
            if verbose == 1:
                print('Deleted existing file.')
    return pathfile


mod = 'QAM16'
filename = 'DetNet_' + mod + \
    '_{}_{}_{}_snr{}_{}'.format(K2, N2, L, ebn0db_low, ebn0db_high) + fn_ext

path_curves = os.path.join('curves', mod, '{}x{}'.format(K, N))
path_models = os.path.join('models_cmdnet', mod, '{}x{}'.format(K, N), filename)

# Save model for import in own script: for detailed evaluation
pathfile = os.path.join(save_dir, path_models, filename)  # + '.ckpt'
check_path(pathfile, verbose=1)


# Training DetNet
total_time = 0
start_time = tm.time()
for i in range(train_iter_no_noise):  # num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I, x_ind = generate_data_train(
        train_batch_size, K, N, snr_low, snr_high, 0, compl, rho, Phi_R12)
    train_step1.run(
        feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X, X_IND: x_ind})
    if i % 1000 == 0:
        # sys.stderr.write(str(i)+' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I, x_ind = generate_data_iid_test(
            train_batch_size, K, N, snr_low, snr_high, 0, compl, rho, Phi_R12)
        results = sess.run(
            [LOSS[L-1], SER], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND: x_ind})
        # print_string = [i]+results
        # print(' '.join('%s' % x for x in print_string))
        train_time = tm.time() - start_time
        total_time = total_time + train_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        print_str = "Epoch: {}, Val Loss: {:.6f}, SER: {:.2e}, Time: {:04.2f}s, Tot. time: {}:{:02d}:{:02d}:{:02d}".format(
            i, results[0], results[1], train_time, int(d), int(h), int(m), int(np.round(s)))
        print(print_str)
        save_path = saver.save(sess, pathfile)  # , global_step = train_iter
        start_time = tm.time()

for i in range(train_iter):  # num of train iter
    batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I, x_ind = generate_data_train(
        train_batch_size, K, N, snr_low, snr_high, 1, compl, rho, Phi_R12)
    train_step2.run(
        feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X, X_IND: x_ind})
    if i % 1000 == 0:
        # sys.stderr.write(str(i)+ ' ')
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I, x_ind = generate_data_iid_test(
            train_batch_size, K, N, snr_low, snr_high, 1, compl, rho, Phi_R12)
        results = sess.run(
            [LOSS[L-1], BER[L-1]], {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND: x_ind})
        # print_string = [i]+results
        # print(' '.join('%s' % x for x in print_string))
        train_time = tm.time() - start_time
        total_time = total_time + train_time
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        print_str = "Epoch: {}, Val Loss: {:.6f}, SER: {:.2e}, Time: {:04.2f}s, Tot. time: {}:{:02d}:{:02d}:{:02d}".format(
            i, results[0], results[1], train_time, int(d), int(h), int(m), int(np.round(s)))
        print(print_str)
        save_path = saver.save(sess, pathfile)  # , global_step = train_iter
        start_time = tm.time()

# saver.restore(sess, "./DetNet_HD_16QAM/16QAM_HD_model.ckpt")
# Final save
save_path = saver.save(sess, pathfile)  # , global_step = train_iter


# Testing the trained model
snrdb_list = np.linspace(snrdb_low_test, snrdb_high_test, num_snr)
snr_list = 10.0 ** (snrdb_list/10.0)
bers = np.zeros((1, num_snr))
times = np.zeros((1, num_snr))
tmp_bers = np.zeros((1, test_iter))
tmp_times = np.zeros((1, test_iter))
for j in range(num_snr):
    for jj in range(test_iter):
        print('snr:')
        print(snrdb_list[j])
        print('test iteration:')
        print(jj)
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1, H_R, H_I, x_R, x_I, w_R, w_I, x_ind = generate_data_iid_test(
            test_batch_size, K, N, snr_list[j], snr_list[j], 1, compl, rho, Phi_R12)

        tic = tm.time()
        tmp_bers[0, jj] = np.array(
            sess.run(SER, {HY: batch_HY, HH: batch_HH, X: batch_X, X_IND: x_ind}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic

        bers[:, j] = np.mean(tmp_bers, 1)
        times[:, j] = np.mean(tmp_times[0])/test_batch_size

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)

# save_path = saver.save(sess, "./DetNet_HD_16QAM/16QAM_HD_model.ckpt")
# Last save for quick comparison
EbN0 = snrdb_list - snr_shift
pathfile = os.path.join(save_dir, path_curves, 'BER_' + filename + '.npz')
check_path(pathfile, verbose=0)
np.savez(pathfile, ebn0=EbN0, ber=bers[0, :])

# Close session to free memory
# sess.close()
