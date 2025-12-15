#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""
import numpy as np
from tensorflow.keras.utils import to_categorical as konehot
import cvxpy as cp         # only for SDR detector
import my_math_operations as mop
import my_communications as com


# MIMO Equalizer/Detector---------------------------------------------------

# TODO: MAP detector

def ml_detector(yr, H, m, res='float64'):
    '''Maximum Likelihood detector for uncoded transmission through channel based on distance measure
    INPUT:
    yr: Received signal of dimension (N_batch, Nr)
    H: Channel matrix (N_batch, Nr, Nt)
    m: Symbol alphabet
    OUTPUT:
    cl: Symbol classes
    '''
    Nt = H.shape[-1]
    M = m.shape[0]
    k = int(Nt * np.log2(M))
    # Possible symbol sequences
    poss_seq = mop.int2bin(np.arange(2 ** k), k)
    poss_seqi = com.modulator(poss_seq.reshape(
        (-1, Nt, int(np.log2(M)))), m).astype('int')
    # Calculation of all possible euclidian distances
    dist = euclid_distH(yr.astype(res), poss_seqi.astype(res), H.astype(res))
    # Vector classes: Actual output of ML detector
    cl_vec = np.argmin(dist, axis=-1)
    # Transform to symbol classes inside vector
    cl = mop.bin2int(poss_seq[cl_vec, :].reshape(
        (dist.shape[0], -1, int(np.log2(M)))))
    fr = konehot(cl, num_classes=M)
    x = m[cl]
    return x, fr, cl


def ml_decoder(yr, G, Lambda, m):
    '''Maximum Likelihood decoding for coded transmission based on distance measure
    INPUT:
    yr: Received signal of dimension (N_batch, n / log2(M))
    G: Generator matrix for code
    Lambda: Covariance matrix
    m: Symbol alphabet
    OUTPUT:
    ur: Estimated receive bits
    '''
    M = m.shape[0]
    k = G.shape[0]
    poss_seq = mop.int2bin(np.arange(2 ** k), k)
    # Possible code sequences
    poss_cseq = com.encoder(poss_seq, G)
    poss_seqi = com.modulator(poss_cseq.reshape(
        (poss_cseq.shape[0], -1, int(np.log2(M)))), m)
    # Calculation of all possible euclidian distances
    dist = euclid_dist(yr, poss_seqi, Lambda)
    cl_vec = np.argmin(dist, axis=-1)
    ur = poss_seq[cl_vec, :]
    return ur


def euclid_distH(A, B, H):
    '''Calculate squared euclidian distance between all rows of A and B with different channels H
    '''
    return np.sum(np.abs(A[:, :, np.newaxis] - np.einsum('inj,kj->ink', H, B)) ** 2, axis=1)


def euclid_dist(A, B, Lambda):
    '''Calculate squared euclidian distance between all rows of A and B
    Also quadratic forms in general can be considered with Lambda = Sigma ** -1
    '''
    Lambda = np.array(Lambda)
    if Lambda.shape != ():
        dist = np.abs(np.sum((A[:, np.newaxis] - B).conj() * (np.dot(A,
                      Lambda.T)[:, np.newaxis] - np.dot(B, Lambda.T)), axis=-1))
    else:
        # 2 fast solutions
        # dist = np.sum(np.abs( A[:, np.newaxis] - B) ** 2, axis=-1)
        dist = (np.abs(A) ** 2).sum(axis=-1)[:, np.newaxis] + (np.abs(B) ** 2).sum(
            axis=-1) - 2 * np.real(np.squeeze(A.dot(np.conj(B)[..., np.newaxis]), axis=-1))
    return dist


# Linear detectors ----------------------------------------------------------


def mmse(data):
    '''Computes Minimum Mean Square Error Solution x for batches of matrices H and observation vectors y in list data
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    OUTPUT
    x_est: Equalized symbols according to mmse solution 
    Phi_ee: Covariance matrix of error in x_est
    '''
    y = data[0]
    H = data[1]
    sigma = data[2]
    # 1. MMSE solution
    HHHI = np.linalg.inv(mop.batch_dot(np.conj(np.transpose(H, (0, 2, 1))), H) +
                         sigma[:, np.newaxis, np.newaxis] ** 2 / 1 * np.eye(H.shape[-1])[np.newaxis, :, :])
    G_MMSE = mop.batch_dot(HHHI, np.conj(np.transpose(H, (0, 2, 1))))
    HSigmayH = mop.batch_dot(G_MMSE, H)
    x_est = mop.batch_dot(G_MMSE, y) / \
        mop.tdiag2vec(HSigmayH)  # unbiased estimator
    # Diagonal of error covariance matrix -> Soft information
    Phi_ee = mop.tvec2diag(1 / mop.tdiag2vec(HSigmayH) - 1)
    # Full error covariance matrix
    # G = batch_dot(tvec2diag(1 / tdiag2vec(HSigmayH)), G_MMSE)
    # GHI = batch_dot(G, H) - np.eye(G.shape[-1])
    # Phi_ee_full = batch_dot(GHI, np.conj(np.transpose(GHI, (0, 2, 1)))) + sigma[:, np.newaxis, np.newaxis] ** 2 * batch_dot(G, np.conj(np.transpose(G, (0, 2, 1))))
    return x_est, Phi_ee


def ls_sol(data):
    '''Computes Least Squares Solution x for batches of matrices H and observation vectors y in list data
    + Calculation of soft information for LS solution
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    OUTPUT
    x_est: Equalized symbols by zero forcer
    Phi_ee: Covariance matrix of error in x_est
    '''
    y = data[0]
    H = data[1]
    sigma = data[2]
    # 2. LS solution
    G = np.linalg.pinv(H)
    x_est = mop.batch_dot(G, y)
    # Soft information (No interference)
    Phi_ee = sigma[:, np.newaxis, np.newaxis] ** 2 * \
        np.linalg.inv(mop.batch_dot(np.conj(np.transpose(H, (0, 2, 1))), H))
    return x_est, Phi_ee


def matched_filter(data):
    '''Computes Matched Filter Solution x for batches of matrices H and observation vectors y in list data
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    OUTPUT
    x_est: Equalized symbols by matched filter
    Phi_ee: Covariance matrix of error in x_est
    '''
    y = data[0]
    H = data[1]
    sigma = data[2]
    # 0. Matched filter solution
    x_est = mop.batch_dot(np.conj(np.transpose(H, (0, 2, 1))), y)
    # Soft information
    HH = mop.batch_dot(np.conj(np.transpose(H, (0, 2, 1))), H)
    Phi_ee = mop.batch_dot(
        HH, HH) + (sigma[:, np.newaxis, np.newaxis] ** 2 - 2) * HH + np.eye(HH.shape[-1])
    return x_est, Phi_ee


def lin_det_soft(x_est, Phi_ee, m, alpha):
    '''Calculation of soft information of symbols from linear detectors
    INPUT
    x_est: Equalized symbols by linear equalizer
    Phi_ee: Covariance matrix of error in x_est
    m: modulation alphabet
    alpha: modulation probabilities
    OUTPUT
    p_x: symbol probabilities
    '''
    # Neglection of non-diagonal entries of Sigma
    arg = - 1 / 2 / mop.tdiag2vec(Phi_ee)[..., np.newaxis] * (
        x_est[..., np.newaxis] - m[np.newaxis, np.newaxis, ...]) ** 2 + np.log(alpha[np.newaxis, ...])
    p_x = mop.np_softmax(arg)
    return p_x


# Nonlinear detectors ----------------------------------------------------------


def sdrSolver(hBatch, yBatch, constellation):
    '''SDR equalizer
    hBatch: Batch of channels H
    ybatch: Batch of received signals y
    constellation: symbol vector
    '''
    results = []
    NT = hBatch.shape[-1]
    for i, H in enumerate(hBatch):
        y = yBatch[i]
        s = cp.Variable((NT, 1), complex=False)
        S = cp.Variable((NT, NT), complex=False)
        objective = cp.Minimize(cp.trace(H.T @ H @ S) - 2. * y.T @ H @ s)
        constraints = [S[i, i] <= (constellation**2).max() for i in range(NT)]
        constraints += [S[i, i] >= (constellation**2).min() for i in range(NT)]
        constraints.append(
            cp.vstack([cp.hstack([S, s]), cp.hstack([s.T, [[1]]])]) >> 0)
        prob = cp.Problem(objective, constraints)
        prob.solve(verbose=False)  # result = prob.solve()
        results.append(s.value)
        # print('SDR it {}'.format(i))
    results = np.array(results)[:, :, 0]
    # print(results.shape)
    fr = konehot(np.argmin(np.abs(results[:, :, np.newaxis] - constellation[np.newaxis,
                 np.newaxis, :]) ** 2, axis=-1), num_classes=constellation.shape[0])
    return fr, results


def mfvi(data, mod, it, seq=1, binopt=0, noise_est=0):
    '''Linear channel equalizer based on mean-field variational inference (mfvi)
    --------------------------------------------------
    INPUT
    H: Real channel matrix
    y: Received signal
    sigma: noise standard deviation
    pi: a-priori probabilities
    it: number of iterations
    seq: sequential or parallel updates (sequential -> guarantees decrease of free energy)
    OUTPUT
    s: estimated symbols
    gam: estimated prob of symbols
    '''
    def mfvinference_bin(data, mod, it, seq=1, noise_est=0):
        '''Inference based on mean-field variational inference
        Implementation according to "A Variational Inference Framework for Soft-In Soft-Out Detection in Multiple-Access Channels" - DD Lin et al., 2009, pp. 2355 (11)
        '''
        y = data[0]
        H = data[1]
        if noise_est == 0:
            N0 = data[2] ** 2
        if (mod.m == np.array([-1, 1])).all():
            pi = mod.alpha[:, 0]
        else:
            pi = mod.alpha[:, 1]
        LLRa = np.log(pi / (1 - pi))
        # Starting point
        gam0 = pi  # a-priori
        gam = np.repeat(gam0[np.newaxis, :], y.shape[0], axis=0)
        HH = mop.batch_dot(np.transpose(H, (0, 2, 1)), H)
        B = HH * (np.ones(HH.shape) -
                  np.expand_dims(np.eye(HH.shape[-1]), 0))  # non-diag HH
        # mf.batch_dot(np.ones((H.shape[0], H.shape[-1])), B)
        B1 = np.sum(B, axis=-1)
        yH = mop.batch_dot(y, H)

        for _ in range(0, it):
            # Joint variance estimation
            if noise_est == 1:
                s = (1 - 2 * gam)
                N0 = np.mean((y - mop.batch_dot(H, s)) ** 2, axis=-1) + \
                    np.mean(mop.tdiag2vec(HH) * (1 - s ** 2), axis=-1)
            if seq == 1:
                # sequential updates -> guarantees decrease of free energy
                for k in range(0, pi.shape[0]):
                    gam[:, k] = mop.sigmoid(
                        LLRa[k] - 2 / N0 * (yH[:, k] - B1[:, k] + 2 * np.sum(gam * B[:, :, k], axis=-1)))
            else:
                # parallel updates -> computational favorable
                gam = mop.sigmoid(
                    LLRa - 2 / N0[:, np.newaxis] * (yH - B1 + 2 * mop.batch_dot(B, gam)))

        s = (1 - 2 * gam)
        if (mod.m == np.array([-1, 1])).all():
            p_x = np.concatenate(
                [gam[:, :, np.newaxis], 1 - gam[:, :, np.newaxis]], axis=-1)
        else:
            p_x = np.concatenate(
                [1 - gam[:, :, np.newaxis], gam[:, :, np.newaxis]], axis=-1)
        return s, p_x

    def mfvinference(data, mod, it, seq=1, noise_est=0):
        '''Inference based on mean-field variational inference for higher order modulation alphabets
        To be invented.
        '''
        print('Not implemented!')
        s = 0
        p_x = 0
        return s, p_x

    if mod.M == 2 and binopt == 1:
        s, p_x = mfvinference_bin(data, mod, it, seq, noise_est=noise_est)
    else:
        s, p_x = mfvinference(data, mod, it, seq, noise_est=noise_est)
    return s, p_x


def AMP(data, mod, it, binopt=0):
    '''Approximate Message Passing Algorithm (AMP)
    Implementation according to "Optimal Detection in Large MIMO" - Jeon et al., 2018, pp. 38 / 14
    --------------------------------------------------
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    mod.alpha: Prior probabilities
    mod.m: Modulation alphabet
    it: number of iterations
    binopt: Select special binary case
    OUTPUT
    s: estimated symbols
    w_m: estimated prob of symbols, one-hot vectors
    '''
    def AMP_bin(data, mod, it):
        '''AMP for binary modulation alphabet m = [-1, 1] / [1, -1]
        Implementation according to "Optimal Detection in Large MIMO" - Jeon et al., 2018, pp. 38
        '''
        yt = data[0]  # .astype(dtype = 'float128')
        Ht = data[1]
        N0 = data[2] ** 2
        m = mod.m
        alpha = mod.alpha
        beta = Ht.shape[-1] / Ht.shape[-2]
        # Starting point
        s = np.dot(alpha, m)[np.newaxis, :]  # a-priori mean
        tau = beta * np.mean(np.dot(alpha, m ** 2)) / N0
        HH = mop.batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = mop.batch_dot(yt, Ht)
        rH = yH - mop.batch_dot(HH, s)
        for _ in range(0, it):
            z = s + rH
            var_F = N0 * (1 + tau)
            s = np.tanh(z / var_F[:, np.newaxis])
            tau_old = tau
            tau = beta / N0 * np.mean(1 - s ** 2, axis=-1)
            rH = yH - mop.batch_dot(HH, s) + \
                (tau / (tau_old + 1))[:, np.newaxis] * rH

        if (m == np.array([-1, 1])).all():
            w_m = np.concatenate(
                [(1 - s[:, :, np.newaxis]) / 2, (1 + s[:, :, np.newaxis]) / 2], axis=-1)
        else:
            w_m = np.concatenate(
                [(1 + s[:, :, np.newaxis]) / 2, (1 - s[:, :, np.newaxis]) / 2], axis=-1)
        return s, w_m

    def AMP_multiclass(data, mod, it):
        '''AMP for higher order modulation alphabets
        Implementation according to "Optimal Detection in Large MIMO" - Jeon et al., 2018, pp. 14
        '''
        yt = data[0]  # .astype(dtype = 'float128')
        Ht = data[1]
        N0 = data[2] ** 2
        m = mod.m
        alpha = mod.alpha
        beta = Ht.shape[-1] / Ht.shape[-2]
        # Starting point
        s = np.dot(alpha, m)[np.newaxis, :]  # a-priori mean
        tau = beta * np.mean(np.dot(alpha, m ** 2)) / N0
        HH = mop.batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = mop.batch_dot(yt, Ht)
        rH = yH - mop.batch_dot(HH, s)
        for _ in range(0, it):
            z = s + rH
            var_F = N0 * (1 + tau)
            arg = - 1 / 2 / var_F[:, np.newaxis, np.newaxis] * (
                z[:, :, np.newaxis] - m[np.newaxis, np.newaxis, :]) ** 2 + np.log(alpha[np.newaxis, :, :])
            w_m = mop.np_softmax(arg)
            s = np.sum(w_m * m[np.newaxis, np.newaxis, :], axis=-1)
            G = np.sum(w_m * (m[np.newaxis, np.newaxis, :] -
                       s[:, :, np.newaxis]) ** 2, axis=-1)
            tau_old = tau
            tau = beta / N0 * np.mean(G, axis=-1)
            rH = yH - mop.batch_dot(HH, s) + \
                (tau / (tau_old + 1))[:, np.newaxis] * rH
        return s, w_m

    if mod.M == 2 and binopt == 1:  # optional AMP_bin
        s, w_m = AMP_bin(data, mod, it)
    else:
        s, w_m = AMP_multiclass(data, mod, it)
    return s, w_m


def np_CMD(data, mod, it, delta, taui, binopt=0):
    '''Inference based on gumbel-softmax distribution -> non-convex and no analytical solution -> Gradient descent solution
    --------------------------------------------------------
    INPUT
    data: list of [yt, Ht, sigma] of MIMO system
    mod.alpha: Prior probabilities
    mod.m: Modulation alphabet
    it: Number of iterations
    taui: Inverse of softmax temperature (size of iterations + 1)
    delta: Gradient step size (size of iterations)
    binopt: Select special binary case
    OUTPUT
    xt: estimated symbols
    ft: estimated prob of symbols, one-hot vectors
    '''
    def np_CMD_bin(data, mod, it, delta, tau):
        '''Binary case with modulation alphabet m = [-1, 1] / [1, -1]
        alpha: Prior probabilities of -1
        '''
        yt = data[0]
        Ht = data[1]
        sigmat = data[2][:, np.newaxis]  # scalar?
        m = mod.m
        if (m == np.array([-1, 1])).all():
            alpha = mod.alpha[:, 0]
        else:
            alpha = mod.alpha[:, 1]
        # Gumbel softmax problem solved with gradient descent
        # Starting point
        s0 = 0 * alpha  # a-priori
        s = np.repeat(s0[np.newaxis, :], Ht.shape[0], axis=0)
        HH = mop.batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = mop.batch_dot(yt, Ht)
        for iteration in range(0, it):
            # functional iteration
            xt = np.tanh((np.log(1 / alpha - 1) + s) / (2 * tau[iteration]))
            xHH = mop.batch_dot(xt, HH)
            grad_L = 1 / (2 * tau[iteration]) * (1 - xt ** 2) * \
                (xHH - yH) + sigmat ** 2 * np.tanh(s / 2)
            s = s - delta[iteration] * grad_L
        # Final evaluation of transmitted symbols
        xt = np.tanh((np.log(1 / alpha - 1) + s) / (2 * tau[-1]))

        if (m == np.array([-1, 1])).all():
            ft = np.concatenate(
                [(1 - xt[:, :, np.newaxis]) / 2, (1 + xt[:, :, np.newaxis]) / 2], axis=-1)
        else:
            ft = np.concatenate(
                [(1 + xt[:, :, np.newaxis]) / 2, (1 - xt[:, :, np.newaxis]) / 2], axis=-1)
        return ft, xt

    def np_CMD_multiclass(data, mod, it, delta, tau):
        '''Multiclass
        '''
        yt = data[0]
        Ht = data[1]
        sigmat = data[2][:, np.newaxis, np.newaxis]  # scalar?
        alpha = mod.alpha
        m = mod.m[np.newaxis, np.newaxis, :]
        # Gumbel softmax problem solved with gradient descent
        # Starting point
        G0 = np.zeros((Ht.shape[-1], m.shape[-1]))  # A-priori
        G = np.repeat(G0[np.newaxis, :], Ht.shape[0], axis=0)
        HH = mop.batch_dot(np.transpose(Ht, (0, 2, 1)), Ht)
        yH = mop.batch_dot(yt, Ht)
        for iteration in range(0, it):
            # functional iteration
            arg = (np.log(alpha) + G) / tau[iteration]
            ft = mop.np_softmax(arg, -1)
            xt = np.sum(ft * m, axis=-1)
            xHH = mop.batch_dot(xt, HH)
            grad_x = 1 / tau[iteration] * (ft * m - ft * xt[:, :, np.newaxis])
            grad_L = grad_x * \
                (xHH - yH)[:, :, np.newaxis] + sigmat ** 2 * (1 - np.exp(-G))
            G = G - delta[iteration] * grad_L
        # Final evaluation of transmitted symbols
        arg = (np.log(alpha) + G) / tau[-1]
        ft = mop.np_softmax(arg, -1)
        xt = np.sum(ft * m, axis=-1)
        return ft, xt

    if mod.M == 2 and binopt == 1:
        ft, xt = np_CMD_bin(data, mod, it, delta, 1 / taui)
    else:
        ft, xt = np_CMD_multiclass(data, mod, it, delta, 1 / taui)

    return ft, xt

# EOF
