#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 16:49:30 2019

@author: beck
"""

import numpy as np
import scipy as sp
import scipy.integrate as integrate
from tensorflow.keras.utils import to_categorical as konehot

import utilities.my_math_operations as mop
import utilities.my_communications as com


# Channel Simulation -------------------------------------------------------


def mimo_channel(Nb, Nr, Nt, compl=0):
    '''Generate [Nb] MIMO channel matrices H with [Nr]x[Nt] Gaussian/Rayleigh Fading taps
    compl: Complex (Rayleigh) or real-valued (Gaussian)
    '''
    if compl == 1:
        H = (np.random.normal(0, 1, (Nb, Nr, Nt)) + 1j *
             np.random.normal(0, 1, (Nb, Nr, Nt))) / np.sqrt(2 * Nr)
    else:
        H = np.random.normal(0, 1, (Nb, Nr, Nt)) / np.sqrt(Nr)
    return H


def mimo_channel_corr(Nb, Nr, Nt, compl=0, rho=0, mode=0):
    '''Generate [Nb] correlated MIMO channel matrices H with [Nr]x[Nt] Rayleigh Fading taps
    - According to Dirk's dissertation for a Uniform Linear Array
    compl: Complex or real-valued
    rho: Correlation
    mode: Correlation only at receiver (0, default) or both transmitter and receiver (1)
    '''
    # Channel matrix w/o correlations
    H_w = mimo_channel(Nb, Nr, Nt, compl)  # * np.sqrt(Nr)
    if rho == 0:
        H = H_w
    else:
        if mode == 0:
            # Version 2: Correlation only at receiver (default)
            phi_row = rho ** (np.arange(Nr) ** 2)
            Phi_R = sp.linalg.toeplitz(phi_row, phi_row)
            Phi_R12 = sp.linalg.fractional_matrix_power(Phi_R, 0.5)
            H = mop.batch_dot(Phi_R12[np.newaxis, :, :], H_w)
            # H = H / np.sqrt(Nr)       # Debugging
        else:
            # Version 1: Correlation at receiver and transmitter
            # Correlation matrix at transmitter
            phi_row = rho ** (np.arange(Nt) ** 2)
            Phi_T = sp.linalg.toeplitz(phi_row, phi_row)
            Phi_T12 = sp.linalg.fractional_matrix_power(Phi_T, 0.5)
            # Correlation matrix at receiver
            phi_row = rho ** (np.arange(Nr) ** 2)
            Phi_R = sp.linalg.toeplitz(phi_row, phi_row)
            Phi_R12 = sp.linalg.fractional_matrix_power(Phi_R, 0.5)
            # Compute correlated channel matrix
            H = mop.batch_dot(mop.batch_dot(
                Phi_R12[np.newaxis, :, :], H_w), Phi_T12[np.newaxis, :, :])

    # Debugging code
    # Test for correctness of implementation
    # Phi_H = np.kron(Phi_T, Phi_R)
    # H_vec = np.transpose(H, (0, 2, 1)).reshape((H.shape[0], -1))
    # Phi_H2 = np.mean(np.einsum('ij,ik->ijk', H_vec, np.conj(H_vec)), axis = 0)
    # # Same as Phi_T and Phi_R up to scaling factor...
    # Phi_T2 = np.mean(mf.batch_dot(np.conj(np.transpose(H, (0, 2, 1))), H), axis = 0)
    # Phi_R2 = np.mean(mf.batch_dot(H, np.conj(np.transpose(H, (0, 2, 1)))), axis = 0)

    # Compare with alternative exact computation -> same
    # Phi_H = np.kron(Phi_T, Phi_R)
    # Phi_H12 = sp.linalg.fractional_matrix_power(Phi_H, 0.5)
    # H_w_vec = np.transpose(H_w, (0, 2, 1)).reshape((H_w.shape[0], -1))
    # H_vec2 = mf.batch_dot(Phi_H12[np.newaxis, : , :], H_w_vec)
    # Phi_H3 = np.mean(np.einsum('ij,ik->ijk', H_vec2, np.conj(H_vec2)), axis = 0)
    return H


def mimo_channel_onering(Nb, Nr, Nt, Phi_R12, compl=0):
    '''Generate [Nb] correlated MIMO channel matrices H with [Nr]x[Nt] Rayleigh Fading taps
    - According to one ring model of Massive MIMO Book!
    Phi_R12: Square root of [Nr]x[Nr] channel covariance matrix at receiver side antenna array
    compl: Complex or real-valued
    '''
    # Channel matrix w/o correlations
    H_w = mimo_channel(Nb, Nr, Nt, compl)
    H = np.zeros(H_w.shape, dtype='complex128')
    # Sampling without replacement
    theta_ind = np.array([np.random.choice(
        range(0, Phi_R12.shape[0]), (Nt), replace=0) for _ in range(Nb)])
    H = np.einsum('nmij,njm->nim', Phi_R12[theta_ind, :, :], H_w)
    # Iterative implementation - slower
    # theta_ind = np.random.randint(0, Phi_R12.shape[0], (Nb, Nt))
    # for ii in range(0, Nt):
    #      # Compute correlated channel matrix column
    #      H[:, :, ii] = mop.batch_dot(Phi_R12[theta_ind[:, ii], :, :], H_w[:, :, ii])
    return H


def mimo_OneRingModel(N, angularSpread, cell_sector=360, compl=1, D=1/2):
    '''This is an implementation of the channel covariance matrix with the
    one-ring model converted from Matlab to Python. The implementation is
    based on Eq. (57) in the paper:

    A. Adhikary, J. Nam, J.-Y. Ahn, and G. Caire, “Joint spatial division and
    multiplexing—the large-scale array regime,” IEEE Trans. Inf. Theory,
    vol. 59, no. 10, pp. 6441-6463, 2013.

    The Matlab version is used in the article:

    Emil Björnson, Jakob Hoydis, Marios Kountouris, Mérouane Debbah, “Massive
    MIMO Systems with Non-Ideal Hardware: Energy Efficiency, Estimation, and
    Capacity Limits,” To appear in IEEE Transactions on Information Theory.

    Download article: http://arxiv.org/pdf/1307.2584

    Based on version 1.0 (Last edited: 2014-08-26)

    License: This code is licensed under the GPLv2 license. If you in any way
    use this code for research that results in publications, please cite our
    original article listed above.

    INPUT
    N: Number of antennas
    angularSpread: Angular spread around the main angle of arrival, e.g., (10, 20)
    theta_grad: Angle of arrival (in degree)
    cell_sector: Cell sector/possible angles of arrival (in degree)
    compl: Complex-valued system model?
    D: Antenna distance in wavelengths (=0.5)
    OUTPUT
    R: [N]x[N] channel covariance matrix
    R12: Square root of R
    '''
    # Define integrand of Eq. (57) in [42]
    def F(alpha, D, distance, theta, Delta):
        return np.exp(-1j * 2 * np.pi * D * distance * np.sin(alpha + theta)) / (2 * Delta)
    # def complex_integrate(func, a, b, **kwargs):
    #     def real_func(x):
    #         return np.real(func(x))
    #     def imag_func(x):
    #         return np.imag(func(x))
    #     real_integral = integrate.quad(real_func, a, b, **kwargs)
    #     imag_integral = integrate.quad(imag_func, a, b, **kwargs)
    #     return (real_integral[0] + 1j*imag_integral[0], real_integral[1:], imag_integral[1:])
    # Approximated angular spread
    Delta = angularSpread * np.pi / 180
    # Half a wavelength distance
    # D = 1 / 2
    # Angle of arrival (30 degrees)
    # theta = theta_grad * np.pi / 180 # np.pi / 6
    # The covariance matrix has the Toeplitz structure, so we only need to
    # compute the first row.
    firstRow = np.zeros((cell_sector, N), dtype='complex128')
    R12 = np.zeros((cell_sector, N, N), dtype='complex128')

    # Go through all columns in the first row
    for theta_grad in range(-int(cell_sector / 2), int(cell_sector / 2)):
        theta = theta_grad * np.pi / 180
        for col in range(0, N):
            # Distance from the first antenna
            distance = col
            # Compute the integral as in [42]
            re = integrate.quad(lambda alpha: np.real(
                F(alpha, D, distance, theta, Delta)), -Delta, Delta)[0]
            if compl == 0:
                # TODO: Exact computation of Phi_R12 for real-valued channels?
                firstRow[theta_grad, col] = np.real(re)
            else:
                im = integrate.quad(lambda alpha: np.imag(
                    F(alpha, D, distance, theta, Delta)), -Delta, Delta)[0]
                firstRow[theta_grad, col] = re + 1j * im
        R12[theta_grad, :, :] = sp.linalg.fractional_matrix_power(
            sp.linalg.toeplitz(firstRow[theta_grad, :]), 0.5)
    # Compute the covarince matrix by utilizing the Toeplitz structure
    # R = sp.linalg.toeplitz(firstRow)
    # Normalization with tr(R)=Nr already included
    return R12


def generate_channel(Nb, Nr, Nt, compl=0, rho=0):
    '''Generates complex or real-valued [Nr]x[Nt] channel according to correlation model (of a Uniform Linear Array)
    Nb: Number of channel realizations
    Nr: Effective number of receive antennas
    Nt: Effective number of transmit antennas
    compl: Complex (1) / real (0)
    rho: Correlation parameter - iid Gaussian (0) / Uniform linear array (0-1) / One ring model (>1)
    '''
    if compl == 1:
        H = mimo_channel_corr(Nb, int(Nr / 2), int(Nt / 2), compl, rho)
        Hr = mop.matim2re(H, 1)
    else:
        Hr = mimo_channel_corr(Nb, Nr, Nt, compl, rho)
    return Hr


def generate_channel_onering(Nb, Nr, Nt, Phi_R12, compl=0):
    '''Generates complex or real-valued [Nr]x[Nt] channel according to one ring correlation model
    Nb: Number of channel realizations
    Nr: Number of receive antennas
    Nt: Number of transmit antennas
    Phi_R12: Square root of [Nr]x[Nr] channel covariance matrix at receiver side antenna array
    compl: Complex (1) / real (0)
    '''
    if compl == 1:
        H = mimo_channel_onering(Nb, int(Nr / 2), int(Nt / 2), Phi_R12, compl)
        Hr = mop.matim2re(H, 1)
    else:
        Hr = mimo_channel_onering(Nb, Nr, Nt, Phi_R12, compl)
    return Hr


class dataset_gen():
    '''Data object for generic data set generation with fixed transmitter and MIMO channel
    data = [y, Hr, sigma, z, x]
    '''
    # Class Attribute
    name = 'MIMO dataset generator'
    # Initializer / Instance Attributes

    def __init__(self, sim_par, soft=0, code=0, G=0, arch=0, online=0):
        '''Input -----------------------------------------
        sim_par: Object from class simulation_parameters
        soft: Decide for classes or symbols as output of data set
        code: Simulation from code words or from symbols? 1/0
        G: Generator matrix if code simulation
        arch: MIMO antenna coding architecture if code simulation
        online: Number of online learning iterations for each noise variance and channel realization (default: 0 for offline training)
        '''
        # Inputs
        self.sim_par = sim_par
        self.soft = soft
        self.online = online    # Number of online iterations, if negative/0 offline training
        self.online_it = 0      # State of online iteration, 0 for offline training
        # Code
        self.code = code
        if self.code == 1:
            self.G = G
            self.arch = arch
            self.intleav = com.random_interleaver()
        # One Ring
        if sim_par.rho[0] > 1:
            if self.sim_par.mod.compl == 1:
                self.Phi_R12 = mimo_OneRingModel(int(
                    sim_par.Nr / 2), sim_par.rho[0], cell_sector=sim_par.rho[1], compl=self.sim_par.mod.compl, D=sim_par.rho[2])
            else:
                self.Phi_R12 = mimo_OneRingModel(
                    sim_par.Nr, sim_par.rho[0], cell_sector=sim_par.rho[1], compl=self.sim_par.mod.compl, D=sim_par.rho[2])
        # Outputs
        self.y = 0
        self.Hr = 0
        self.sigma = 0
        self.z = 0
        self.cl = 0
        self.x = 0
        self.data = []
        self.u = 0
        self.data_val = []      # Important for online training
        self.data_opt = None    # Optimized for fixed input data
        self.N_bval = 0
    # Instance methods

    def __call__(self, Nb=None):
        if self.code == 1:
            return self.gen_coded_mimodata(Nb=Nb)
        elif self.online >= 1:
            self.gen_online_mimodata(Nb=Nb)
        else:
            return self.gen_mimodata(Nb=Nb)

    def gen_mimodata(self, Nb=None):
        '''Generates data according to MIMO channel model for arbitrary symbol alphabet m
        Output ----------------------------------------
        data = [y, Hr, sigma, z, x]
        y: Received signal vector
        Hr: Real-valued channel matrix
        sigma: Standard deviation vector of AWGN
        x: Transmit signal vector
        z: Classes in categorical one-hot encoding
        cl: Class labels
        Input -----------------------------------------
        Nb: Batch size (optional)
        '''
        if Nb == None:
            Nb = self.sim_par.N_batch
        Nr = self.sim_par.Nr
        Nt = self.sim_par.Nt
        # alpha = sim_par.mod.alpha      # Not used
        snr_min = self.sim_par.SNR_range[0]
        snr_max = self.sim_par.SNR_range[1]
        m = self.sim_par.mod.m
        M = self.sim_par.mod.M          # N_class
        compl = self.sim_par.mod.compl
        rho = self.sim_par.rho[0]

        # Generate symbols
        # Generate class labels -----------------------
        self.cl = np.reshape(np.random.randint(0, M, Nb * Nt), (Nb, -1))
        # Convert labels to categorical one-hot encoding
        self.z = konehot(self.cl, num_classes=M)
        self.x = m[self.cl]             # x = np.dot(z, m)

        # Generate Nb2 = Nb channel matrices and SNRs for offline training / Nb2 = 1 for online training
        if rho > 1:
            self.Hr = generate_channel_onering(Nb, Nr, Nt, self.Phi_R12, compl)
        else:
            self.Hr = generate_channel(Nb, Nr, Nt, compl, rho)
        self.sigma = mop.csigma(np.random.uniform(
            snr_min, snr_max, Nb))    # SNR = 7 - 35 dB
        self.y = com.awgn(mop.batch_dot(self.Hr, self.x), np.repeat(
            self.sigma[:, np.newaxis], Nr, axis=-1))

        # Implementation with QR decomposed system model
        # qr = 0
        # if qr == 1:
        #     ## Numpy implementation
        #     # start_time = time.time()
        #     Q = np.zeros((Nb, Nr, min(Nt, Nr)))
        #     R = np.zeros((Nb, min(Nt, Nr), Nt))
        #     for indb in range(0, Nb):
        #         Qb, Rb = np.linalg.qr(self.Hr[indb, :, :], 'reduced')
        #         Q[indb, :, :] = Qb
        #         R[indb, :, :] = Rb
        #     self.Hr = R
        #     self.y = mop.batch_dot(np.transpose(np.conj(Q), (0, 2, 1)), self.y)
        #     ## Tensorflow implementation too slow !!!
        #     # Q, R = tf.linalg.qr(self.Hr)
        #     # self.Hr = KB.eval(R)
        #     # self.y = mop.batch_dot(np.transpose(np.conj(KB.eval(Q)), (0, 2, 1)), self.y)
        #     # print(time.time()- start_time)

        if self.soft == 1:
            self.data = [self.y, self.Hr, self.sigma, self.x]
        elif self.soft == 0:
            self.data = [self.y, self.Hr, self.sigma, self.z]
        else:
            self.data = [self.y, self.Hr, self.sigma, self.z, self.x]

        return self.data

    def gen_online_mimodata(self, Nb=None):
        '''Generates data according to MIMO channel model ONLINE for arbitrary symbol alphabet m and
        realization of channel matrix H and noise std dev sigma
        Output ----------------------------------------
        data = [y, Hr, sigma, z, x]
        y: Received signal vector
        Hr: Real-valued channel matrix
        sigma: Standard deviation vector of AWGN
        x: Transmit signal vector
        z: Classes in categorical one-hot encoding
        cl: Class labels
        Input -----------------------------------------
        Nb: Batch size (optional)
        '''
        if Nb == None:
            Nb = self.sim_par.N_batch
        if self.online >= 1:
            Nb2 = 1
        else:
            Nb2 = Nb
        Nr = self.sim_par.Nr
        Nt = self.sim_par.Nt
        # alpha = sim_par.mod.alpha     # Not used
        snr_min = self.sim_par.SNR_range[0]
        snr_max = self.sim_par.SNR_range[1]
        m = self.sim_par.mod.m
        M = self.sim_par.mod.M          # N_class
        compl = self.sim_par.mod.compl
        rho = self.sim_par.rho[0]

        # Generate symbols
        # Generate class labels -----------------------
        self.cl = np.reshape(np.random.randint(0, M, Nb * Nt), (Nb, -1))
        # Convert labels to categorical one-hot encoding
        self.z = konehot(self.cl, num_classes=M)
        self.x = m[self.cl]             # x = np.dot(z, m)

        # Generate Nb2 = Nb channel matrices and SNRs for offline training / Nb2 = 1 for online training
        if self.online_it == 0:
            if rho > 1:
                self.Hr = generate_channel_onering(
                    Nb2, Nr, Nt, self.Phi_R12, compl)
            else:
                self.Hr = generate_channel(Nb2, Nr, Nt, compl, rho)
            self.sigma = mop.csigma(np.random.uniform(
                snr_min, snr_max, Nb2))   # SNR = 7 - 35 dB
        if self.online >= 1:
            # Note: sigma needs to have the dimension as x to produce the same amount of noise realizations n
            # -> Adaptation of code for just one sigma necessary here
            self.y = com.awgn(mop.batch_dot(self.Hr, self.x),
                              self.sigma * np.ones((Nb, Nr)))
        else:
            self.y = com.awgn(mop.batch_dot(self.Hr, self.x), np.repeat(
                self.sigma[:, np.newaxis], Nr, axis=-1))

        if self.soft == 1:
            self.data = [self.y, self.Hr, self.sigma, self.x]
        elif self.soft == 0:
            self.data = [self.y, self.Hr, self.sigma, self.z]
        else:
            self.data = [self.y, self.Hr, self.sigma, self.z, self.x]

        if self.online >= 1:
            if self.online_it == 0:
                # Generate validation set for current channel matrix and SNR
                # Generate class labels -----------------------
                cl = np.reshape(np.random.randint(
                    0, M, self.N_bval * Nt), (self.N_bval, -1))
                # Convert labels to categorical one-hot encoding
                z = konehot(cl, num_classes=M)
                x = m[cl]
                if self.online >= 1:
                    y = com.awgn(mop.batch_dot(self.Hr, x),
                                 self.sigma * np.ones((self.N_bval, Nr)))
                else:
                    y = com.awgn(mop.batch_dot(self.Hr, x), np.repeat(
                        self.sigma[:, np.newaxis], Nr, axis=-1))
                if self.soft == 1:
                    self.data_val = [y, self.Hr, self.sigma, x]
                elif self.soft == 0:
                    self.data_val = [y, self.Hr, self.sigma, z]
                else:
                    self.data_val = [y, self.Hr, self.sigma, z, x]
                self.data_opt = [self.Hr[0, :, :], self.sigma[0]]

        # Track online learning iteration and reset
        if self.online >= 1:
            self.online_it = self.online_it + 1
            if self.online_it >= self.online:
                self.online_it = 0
        return self.data

    def gen_onlineval_mimodata(self, Nb=None):
        '''Generates data according to MIMO channel model ONLINE for arbitrary symbol alphabet m and
        GIVEN channel matrix H and noise std dev sigma - only for validation 
        Output ----------------------------------------
        data = [y, Hr, sigma, z, x]
        y: Received signal vector
        Hr: Real-valued channel matrix
        sigma: Standard deviation vector of AWGN
        x: Transmit signal vector
        z: Classes in categorical one-hot encoding
        cl: Class labels
        Input -----------------------------------------
        Nb: Batch size (optional)
        '''
        if Nb == None:
            Nb = self.sim_par.N_batch
        Nr = self.sim_par.Nr
        Nt = self.sim_par.Nt
        # alpha = sim_par.mod.alpha     # not used
        m = self.sim_par.mod.m
        M = self.sim_par.mod.M          # N_class

        # Generate validation set for current channel and SNR
        # Generate class labels -----------------------
        self.cl = np.reshape(np.random.randint(0, M, Nb * Nt), (Nb, -1))
        # Convert labels to categorical one-hot encoding
        self.z = konehot(self.cl, num_classes=M)
        self.x = m[self.cl]             # x = np.dot(z, m)
        self.y = com.awgn(mop.batch_dot(self.Hr, self.x),
                          self.sigma * np.ones((Nb, Nr)))
        if self.soft == 1:
            self.data = [self.y, self.Hr, self.sigma, self.x]
        elif self.soft == 0:
            self.data = [self.y, self.Hr, self.sigma, self.z]
        else:
            self.data = [self.y, self.Hr, self.sigma, self.z, self.x]
        return self.data

    def gen_coded_mimodata(self, Nb=None):
        '''Generates coded data according to MIMO channel model for arbitrary symbol alphabet m
        Output ----------------------------------------
        data = [y, Hr, sigma, z, x]
        y: Received signal vector
        Hr: Real-valued channel matrix
        sigma: Standard deviation vector of AWGN
        x: Transmit signal vector
        z: Classes in categorical one-hot encoding
        cl: Class labels
        Input -----------------------------------------
        Nb: Batch size (optional)
        '''
        if Nb == None:
            Nb = self.sim_par.N_batch
        Nr = self.sim_par.Nr
        Nt = self.sim_par.Nt
        # alpha = sim_par.mod.alpha     # Not used
        snr_min = self.sim_par.SNR_range[0]
        snr_max = self.sim_par.SNR_range[1]
        m = self.sim_par.mod.m
        M = self.sim_par.mod.M          # N_class
        compl = self.sim_par.mod.compl
        rho = self.sim_par.rho[0]
        # Code
        arch = self.arch
        G = self.G

        # 1. Generate data bits -----------------------
        Nbc = sym2code_batchsize(Nb, Nt, M, G.shape[-1], arch)
        u = np.random.randint(2, size=(Nbc, G.shape[0]))
        c = com.encoder(u, G)
        self.intleav.shuffle(Nbc, G.shape[-1])
        c_perm = self.intleav.interleave(c)
        # Interface to Equalizer
        c2 = com.mimo_coding(c_perm, Nt, M, arch)

        # 2. Generate class labels -----------------------
        # cl = np.reshape(np.random.randint(0, M, Nb * Nt), (Nb, -1))
        cl = mop.bin2int(c2, axis=-1)
        # Convert labels to categorical one-hot encoding
        z = konehot(cl, num_classes=M).astype('int32')
        if len(z.shape) == 3:
            z = np.expand_dims(z, axis=-2)
        # Generate symbols
        x = m[cl]                       # x = np.dot(z, m)
        Nb2 = x.shape[0]
        # Generate channel matrix
        if rho > 1:
            Hr = generate_channel_onering(Nb2, Nr, Nt, self.Phi_R12, compl)
        else:
            Hr = generate_channel(Nb2, Nr, Nt, compl, rho)
        sigma = mop.csigma(np.random.uniform(
            snr_min, snr_max, Nb2))    # SNR = 7 - 35 dB
        y = com.awgn(mop.batch_dot(Hr, x), sigma[:, np.newaxis].repeat(
            Nr, -1)[:, :, np.newaxis].repeat(x.shape[2], -1))

        # Match dimensions of Equalizer
        cl2 = np.transpose(cl, (0, 2, 1)).reshape((-1, cl.shape[1]))
        z2 = np.transpose(z, (0, 2, 1, 3)).reshape(
            (-1, z.shape[1], z.shape[-1]))
        x2 = np.transpose(x, (0, 2, 1)).reshape((-1, x.shape[1]))
        sigma2 = sigma[:, np.newaxis].repeat(
            x.shape[2], -1).reshape(cl2.shape[0])
        y2 = np.transpose(y, (0, 2, 1)).reshape((-1, y.shape[1]))
        Hr2 = np.expand_dims(Hr, axis=1).repeat(
            x.shape[-1], 1).reshape(-1, Nr, Nt)

        self.cl = cl2
        self.z = z2
        self.x = x2
        self.Hr = Hr2
        self.sigma = sigma2
        self.y = y2
        self.u = u

        if self.soft == 1:
            self.data = [self.y, self.Hr, self.sigma, self.x]
        elif self.soft == 0:
            self.data = [self.y, self.Hr, self.sigma, self.z]
        else:
            self.data = [self.y, self.Hr, self.sigma, self.z, self.x]
        return self.data


def sym2code_batchsize(Nb, Nt, M, n, arch):
    '''Compute batch size Nbc of code words given a batch size Nb of symbol vector x with MIMO encoding arch
    INPUT
    Nb: Batch size of symbol vectors x
    Nt: Length of symbol vector x
    M: Modulation order
    n: Code word length
    arch: Vertical or horizontal MIMO encoding
    OUTPUT
    Nbc: Batch size Nbc of code words
    '''
    if arch == 'horiz':
        # Nt * n is minimum size
        Nb2 = int(np.ceil(Nb / (Nt * np.ceil(n / np.log2(M))))
                  * Nt * np.ceil(n / np.log2(M)))
        Nbc = int(Nb2 * Nt * np.log2(M) / np.ceil(n / np.log2(M)))
    elif arch == 'vert':
        fit2x = Nt * np.log2(M) / n
        if int(fit2x) >= 1:
            Nbc = Nb * int(fit2x)
        else:
            Nbc = int(Nb / np.ceil(1 / fit2x))
    else:
        print('Coding scheme not available!')
    return Nbc


# EOF
