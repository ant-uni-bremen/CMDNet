#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 14:08:50 2019

@author: beck
"""


import os
import tensorflow as tf
import numpy as np
import time as tm

# parameters
K = 60  # 20 Nt * 2
N = 60  # 30 Nr * 2
L = 3 * K  # 90 (Default) #3 * K (VCDN) / K (ShVCDN) (Paper)
# Testing
test_iter = 200
test_batch_size = 1000
num_snr = 18
snrdb_low_test = 4.0
snrdb_high_test = 21.0  # SNR
# Training
do_train = 1
loaded_epoch = 16000
train_iter = 20000  # 20000
train_batch_size = 5000  # 5000
snrdb_low = 7  # 10 - 3 # 7
snrdb_high = 14  # 30 - 3 # 14
snr_low = 10.0 ** (snrdb_low/10.0)
snr_high = 10.0 ** (snrdb_high/10.0)


def generate_channel(B, N, K):
    if K % 2 == 0 and N % 2 == 0:
        H_re = np.random.randn(B, int(N/2), int(K/2))
        H_im = np.random.randn(B, int(N/2), int(K/2))
        # H_ = np.concatenate(((np.concatenate((H_re, -H_im), -1), np.concatenate((H_im, H_re), -1))), 1) / np.sqrt(N)
        H_ = np.concatenate(((np.concatenate(
            (H_re, -H_im), -1), np.concatenate((H_im, H_re), -1))), 1) / np.sqrt(2 * N / 2)
    else:
        H_ = np.random.randn(B, N, K) / np.sqrt(N)
    return H_

# def mimo_channel(Nr, Ntaps, N):
#    '''Generate N MIMO channel matrices with NtapsxNtaps Rayleigh Fading taps
#    '''
#    H = (np.random.normal(0, 1, (N, Nr, Ntaps)) + 1j * np.random.normal(0, 1, (N, Nr, Ntaps))) / np.sqrt(2 * Nr)
#    return H


def generate_data_iid_test(B, K, N, snr_low, snr_high):
    H_ = generate_channel(B, N, K)
    # W_=np.zeros([B,K,K])
    x_ = np.sign(np.random.rand(B, K)-0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_*0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace()/K  # Calculate effective SNR ???
        # H_[i,:,:]=H
        # Adjust noise variance to channel matrix realization ???
        y_[i, :] = (H.dot(x_[i, :])+w[i, :]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        # y_[i,:]=(H.dot(x_[i,:])+w[i,:]/np.sqrt(SNR))
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_


def generate_data_train(B, K, N, snr_low, snr_high):
    H_ = generate_channel(B, N, K)
    # W_=np.zeros([B,K,K])
    x_ = np.sign(np.random.rand(B, K)-0.5)
    y_ = np.zeros([B, N])
    w = np.random.randn(B, N)
    Hy_ = x_*0
    HH_ = np.zeros([B, K, K])
    SNR_ = np.zeros([B])
    for i in range(B):
        SNR = np.random.uniform(low=snr_low, high=snr_high)
        H = H_[i, :, :]
        tmp_snr = (H.T.dot(H)).trace()/K
        # H_[i,:,:]=H
        y_[i, :] = (H.dot(x_[i, :])+w[i, :]*np.sqrt(tmp_snr)/np.sqrt(SNR))
        # y_[i,:]=(H.dot(x_[i,:])+w[i,:]/np.sqrt(SNR))
        Hy_[i, :] = H.T.dot(y_[i, :])
        HH_[i, :, :] = H.T.dot(H_[i, :, :])
        SNR_[i] = SNR
    return y_, H_, Hy_, HH_, x_, SNR_


# Load Detnet

sess = tf.InteractiveSession()  # sess = tf.Session()
pathfile = 'tf_graphs/detnet_{}_{}_{}snr{}_{}/'.format(
    K/2, N/2, L, snrdb_low, snrdb_high)
new_saver = tf.train.import_meta_graph(pathfile+'detnet_{}_{}_{}-'.format(
    K/2, N/2, L, snrdb_low, snrdb_high)+str(loaded_epoch)+'.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(pathfile))


graph = tf.get_default_graph()
BER = graph.get_tensor_by_name("BERt_"+str(L-1)+':0')
# TOTAL_LOSS = graph.get_tensor_by_name("TOTAL_LOSSt:0")
LOSS = graph.get_tensor_by_name("LOSSt_"+str(L-1)+':0')
HY = graph.get_tensor_by_name("HYt:0")
HH = graph.get_tensor_by_name("HHt:0")
X = graph.get_tensor_by_name("Xt:0")


# TRAINING
if do_train == 1:
    train_step = tf.get_collection("train_step")[0]
    saver = tf.train.Saver()
    for i in range(loaded_epoch, train_iter):  # num of train iter
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_train(
            train_batch_size, K, N, snr_low, snr_high)
        train_step.run(feed_dict={HY: batch_HY, HH: batch_HH, X: batch_X})
        if i % 1000 == 0:
            saver.save(sess, pathfile +
                       'detnet_{}_{}_{}'.format(K/2, N/2, L), global_step=i)
        if i % 100 == 0:
            batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(
                train_batch_size, K, N, snr_low, snr_high)
            results = sess.run(
                [LOSS, BER], {HY: batch_HY, HH: batch_HH, X: batch_X})
            print_string = [i]+results
            print(' '.join(f'{x}' for x in print_string))
    saver.save(sess, pathfile+'detnet_{}_{}_{}'.format(K /
               2, N/2, L), global_step=i+1)


# TEST
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
        batch_Y, batch_H, batch_HY, batch_HH, batch_X, SNR1 = generate_data_iid_test(
            test_batch_size, K, N, snr_list[j], snr_list[j])
        tic = tm.time()
        tmp_bers[:, jj] = np.array(
            sess.run(BER, {HY: batch_HY, HH: batch_HH, X: batch_X}))
        toc = tm.time()
        tmp_times[0][jj] = toc - tic
    bers[0][j] = np.mean(tmp_bers, 1)
    times[0][j] = np.mean(tmp_times[0])/test_batch_size

print('snrdb_list')
print(snrdb_list)
print('bers')
print(bers)
print('times')
print(times)


# Save simulation results into file
EbN0 = snrdb_list - 10 * np.log10(2)

pathfile = os.path.join('simulation_results', 'BER_DETNET_{}_{}_{}snr{}_{}.npz'.format(
    K/2, N/2, L, snrdb_low, snrdb_high))
if os.path.isfile(pathfile):
    os.remove(pathfile)
np.savez(pathfile, ebn0=EbN0, ber=bers)
