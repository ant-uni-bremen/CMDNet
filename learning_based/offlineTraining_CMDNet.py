from exp import get_data
import scipy.integrate as integrate
import scipy as sp
import pickle
import argparse
from tf_session import *
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import gen_math_ops
import os
import matplotlib as mpl
mpl.use('Agg')


# Introducing own functions and variables
compl = 1
rho = 10
cell_sector = 120
fn_ext = '_OneRing10_120_snr4_27'


def mimo_channel(Nb, Nr, Nt, compl=0):
    '''Generate [Nb] MIMO channel matrices H with [Nr]x[Nt] Rayleigh Fading taps
    compl: complex or real-valued
    '''
    if compl == 1:
        H = (np.random.normal(0, 1, (Nb, Nr, Nt)) + 1j *
             np.random.normal(0, 1, (Nb, Nr, Nt))) / np.sqrt(2 * Nr)
    else:
        H = np.random.normal(0, 1, (Nb, Nr, Nt)) / np.sqrt(Nr)
    return H


def batch_dot(a, b):
    '''Computes the
    matrix vector product: A*b
    vector matrix product: a*B
    matrix product: A*B
    for a batch of matrices and vectors along dimension 0
    Shape of tensors decides operation
    '''
    if len(a.shape) == 3 and len(b.shape) == 2:
        y = np.einsum('nij,nj->ni', a, b)  # A*b
    elif len(a.shape) == 2 and len(b.shape) == 3:
        y = np.einsum('nj,nji->ni', a, b)  # b*A
    elif len(a.shape) == 3 and len(b.shape) == 3:
        y = np.einsum('nij,njk->nik', a, b)  # A*B
    return y


def mimo_channel_corr(Nb, Nr, Nt, compl=0, rho=0):
    '''Generate [Nb] correlated MIMO channel matrices H with [Nr]x[Nt] Rayleigh Fading taps
    According to Dirk's dissertation for a Uniform Linear Array
    compl: complex or real-valued
    rho: correlation
    '''
    # Channel matrix w/o correlations
    H_w = mimo_channel(Nb, Nr, Nt, compl)  # * np.sqrt(Nr)
    if rho == 0:
        H = H_w
    else:
        # Version 1: Correlation at receiver and transmitter
        # Correlation matrix at transmitter
        # phi_row = rho ** (np.arange(Nt) ** 2)
        # Phi_T = sp.linalg.toeplitz(phi_row, phi_row)
        # Phi_T12 = sp.linalg.fractional_matrix_power(Phi_T, 0.5)
        # # Correlation matrix at receiver
        # phi_row = rho ** (np.arange(Nr) ** 2)
        # Phi_R = sp.linalg.toeplitz(phi_row, phi_row)
        # Phi_R12 = sp.linalg.fractional_matrix_power(Phi_R, 0.5)
        # # Compute correlated channel matrix
        # H = batch_dot(batch_dot(Phi_R12[np.newaxis, :, :], H_w), Phi_T12[np.newaxis, :, :])
        # Version 2: Correlation only at receiver
        phi_row = rho ** (np.arange(Nr) ** 2)
        Phi_R = sp.linalg.toeplitz(phi_row, phi_row)
        Phi_R12 = sp.linalg.fractional_matrix_power(Phi_R, 0.5)
        H = batch_dot(Phi_R12[np.newaxis, :, :], H_w)
        # H = H / np.sqrt(Nr)

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


def matim2re(x, mode=1):
    '''Converts imaginary vector/matrix to real
    '''
    if mode == 1:  # matrix conversion
        if len(x.shape) == 3:
            x = np.concatenate((np.concatenate((np.real(x), np.imag(x)), axis=1), np.concatenate(
                (-np.imag(x), np.real(x)), axis=1)), axis=-1)
        else:
            x = np.concatenate((np.concatenate((np.real(x), np.imag(x))), np.concatenate(
                (-np.imag(x), np.real(x)))), axis=1)
    else:  # vector conversion
        x = np.concatenate((np.real(x), np.imag(x)), axis=1)
    return x


def mimo_channel_onering(Nb, Nr, Nt, Phi_R12, compl=0):
    '''Generate [Nb] correlated MIMO channel matrices H with [Nr]x[Nt] Rayleigh Fading taps
    According to one ring model of Massive MIMO Book
    compl: complex or real-valued
    rho: correlation
    '''
    # Channel matrix w/o correlations
    H_w = mimo_channel(Nb, Nr, Nt, compl)
    H = np.zeros(H_w.shape, dtype='complex128')
    # Correlation matrix at receiver
    # theta_ind = np.random.randint(0, Phi_R12.shape[0], (Nb, Nt))
    # Sampling without replacement
    theta_ind = np.array([np.random.choice(
        range(0, Phi_R12.shape[0]), (Nt), replace=0) for _ in range(Nb)])
    H = np.einsum('nmij,njm->nim', Phi_R12[theta_ind, :, :], H_w)
    # for ii in range(0, Nt):
    #      theta_ind = np.random.randint(0, Phi_R12.shape[0], Nb)
    #      # Compute correlated channel matrix column
    #      H[:, :, ii] = batch_dot(Phi_R12[theta_ind, :, :], H_w[:, :, ii])
    return H


def mimo_OneRingModel(N, angularSpread, cell_sector=360):
    '''This is an implementation of the channel covariance matrix with the
    one-ring model. The implementation is based on Eq. (57) in the paper:

    A. Adhikary, J. Nam, J.-Y. Ahn, and G. Caire, “Joint spatial division and
    multiplexing—the large-scale array regime,” IEEE Trans. Inf. Theory,
    vol. 59, no. 10, pp. 6441–6463, 2013.

    This is used in the article:

    Emil Björnson, Jakob Hoydis, Marios Kountouris, Mérouane Debbah, “Massive
    MIMO Systems with Non-Ideal Hardware: Energy Efficiency, Estimation, and
    Capacity Limits,” To appear in IEEE Transactions on Information Theory.

    Download article: http://arxiv.org/pdf/1307.2584

    This is version 1.0 (Last edited: 2014-08-26)

    License: This code is licensed under the GPLv2 license. If you in any way
    use this code for research that results in publications, please cite our
    original article listed above.

    INPUT
    N: Number of antennas
    angularSpread: Angular spread around the main angle of arrival, e.g., (10, 20)
    theta_grad: Angle of arrival (in degree)
    cell_sector: cell sector/possible angles of arrival (in degree)
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
    D = 1 / 2
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
            im = integrate.quad(lambda alpha: np.imag(
                F(alpha, D, distance, theta, Delta)), -Delta, Delta)[0]
            firstRow[theta_grad, col] = re + 1j * im
        R12[theta_grad, :, :] = sp.linalg.fractional_matrix_power(
            sp.linalg.toeplitz(firstRow[theta_grad, :]), 0.5)
    # Compute the covarince matrix by utilizing the Toeplitz structure
    # R = sp.linalg.toeplitz(firstRow)
    return R12


def generate_channel(Nb, Nr, Nt, compl=0, rho=0, Phi_R12=0):
    '''Generates complex or real-valued [Nr]x[Nt] channel according to one ring correlation model
    Nb: Number of channel realizations
    Nr: Number of receive antennas
    Nt: Number of transmit antennas
    compl: Complex (1) / real (0)
    Phi_R12: correlation matrices
    '''
    if rho > 1:
        if compl == 1:
            H = mimo_channel_onering(
                Nb, int(Nr / 2), int(Nt / 2), Phi_R12, compl)
            Hr = matim2re(H, 1)
        else:
            Hr = mimo_channel_onering(Nb, Nr, Nt, Phi_R12, compl)
    else:
        if compl == 1:
            H = mimo_channel_corr(Nb, int(Nr / 2), int(Nt / 2), compl, rho)
            Hr = matim2re(H, 1)
        else:
            Hr = mimo_channel_corr(Nb, Nr, Nt, compl, rho)
    return Hr
##

# Original script begins


parser = argparse.ArgumentParser(description='MIMO signal detection simulator')

parser.add_argument('--x-size', '-xs',
                    type=int,
                    required=True,
                    help='Number of senders')

parser.add_argument('--y-size', '-ys',
                    type=int,
                    required=True,
                    help='Number of receivers')

parser.add_argument('--layers',
                    type=int,
                    required=True,
                    help='Number of neural net blocks')

parser.add_argument('--snr-min',
                    type=float,
                    required=True,
                    help='Minimum SNR in dB')

parser.add_argument('--snr-max',
                    type=float,
                    required=True,
                    help='Maximum SNR in dB')

parser.add_argument('--learn-rate', '-lr',
                    type=float,
                    required=True,
                    help='Learning rate')

parser.add_argument('--batch-size',
                    type=int,
                    required=True,
                    help='Batch size')

parser.add_argument('--test-every',
                    type=int,
                    required=True,
                    help='number of training iterations before each test')

parser.add_argument('--train-iterations',
                    type=int,
                    required=True,
                    help='Number of training iterations')

parser.add_argument('--modulation', '-mod',
                    type=str,
                    required=True,
                    help='Modulation type which can be BPSK, 4PAM, or MIXED')

parser.add_argument('--gpu',
                    type=str,
                    required=False,
                    default="0",
                    help='Specify the gpu core')

parser.add_argument('--test-batch-size',
                    type=int,
                    required=True,
                    help='Size of the test batch')

parser.add_argument('--data',
                    action='store_true',
                    help='Use dataset to train/test')

parser.add_argument('--linear',
                    type=str,
                    required=True,
                    help='linear transformation step method')

parser.add_argument('--denoiser',
                    type=str,
                    required=True,
                    help='denoiser function model')

parser.add_argument('--exp',
                    type=str,
                    required=False,
                    help='experiment name')

parser.add_argument('--corr-analysis',
                    action='store_true',
                    help='fetch covariance matrices')

parser.add_argument('--start-from',
                    type=str,
                    required=False,
                    default='',
                    help='Saved model name to start from')

parser.add_argument('--log',
                    action='store_true',
                    help='Log data mode')

args = parser.parse_args()
# Ignore if you do not have multiple GPUs
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Own modification
args.data = 1   # activate costum H (!!!)
# Simulation parameters
params = {
    'N': args.y_size,  # Number of receive antennas
    'K': args.x_size,  # Number of transmit antennas
    'L': args.layers,  # Number of layers
    'SNR_dB_min': args.snr_min,  # Minimum SNR value in dB for training and evaluation
    'SNR_dB_max': args.snr_max,  # Maximum SNR value in dB for training and evaluation
    'seed': 1,  # Seed for random number generation
    'batch_size': args.batch_size,
    'modulation': args.modulation,
    #    'TL': args.train_layer,
    'correlation': False,
    #    'save_name': args.saveas,
    'start_from': args.start_from,
    'data': args.data,
    'linear_name': args.linear,
    'denoiser_name': args.denoiser
}

# -- One Ring correlation matrices --------------------
if rho > 1:
    if compl == 1:
        Phi_R12 = mimo_OneRingModel(int(2 * params['N'] / 2), rho, cell_sector)
    else:
        Phi_R12 = mimo_OneRingModel(2 * params['N'], rho, cell_sector)
else:
    Phi_R12 = 0
# -----------------------------------------------------


def complex_to_real(inp):
    Hr = np.real(inp)
    Hi = np.imag(inp)
    h1 = np.concatenate([Hr, -Hi], axis=2)
    h2 = np.concatenate([Hi,  Hr], axis=2)
    inp = np.concatenate([h1, h2], axis=1)
    return inp


# if args.data:
#    H_dataset = np.load('/data/Mehrdad/channel_sequences_normalized.npy')[0,:]
#    H_dataset = np.reshape(H_dataset, (-1, args.y_size, args.x_size))
#    H_dataset = complex_to_real(H_dataset)
#    Hdataset_powerdB = 10. * np.log(np.mean(np.sum(H_dataset ** 2, axis=1))) / np.log(10.)
#    print Hdataset_powerdB
#
#    train_data = H_dataset[0:int(0.8*np.shape(H_dataset)[0])]
#    test_data = H_dataset[int(0.8*np.shape(H_dataset)[0])+1:-1]
  # print "H is fixed" # dont forget to set the above
  # train_data = np.array([H_dataset[0]])
  # test_data = np.array([H_dataset[0]])
# if args.data:
#     train_data_ref, test_data_ref, Hdataset_powerdB = get_data(args.exp)
#     print(train_data_ref.shape)
#     params['Hdataset_powerdB'] = Hdataset_powerdB

# Build the computational graph
mmnet = MMNet_graph(params)
nodes = mmnet.build()

# Get access to the nodes on the graph
sess = nodes['sess']
x = nodes['x']
H = nodes['H']
x_id = nodes['x_id']
constellation = nodes['constellation']
train = nodes['train']
# summary = nodes['summary']
snr_db_min = nodes['snr_db_min']
snr_db_max = nodes['snr_db_max']
lr = nodes['lr']
batch_size = nodes['batch_size']
accuracy = nodes['accuracy']
mmse_accuracy = nodes['mmse_accuracy']
loss = nodes['loss']
# test_summary_writer = nodes['test_summary_writer']
# train_summary_writer = nodes['train_summary_writer']
# saver = nodes['saver']
# train_layer_no = nodes['train_layer_no']
logs = nodes['logs']
measured_snr = nodes['measured_snr']
# Training loop
# tln_ = args.train_layer

# for t in range(20):
#    print "~ meta learning on H"

record = {'before': [], 'after': []}
record_flag = False

# if args.data:
#     train_data = train_data_ref
#     test_data  = test_data_ref
# else:
#     test_data = []
#     train_data = []
for it in range(args.train_iterations):
    # pertr = np.random.normal(0.,0.01, (5000, 64, 16))
    # perti = np.random.normal(0.,0.01, (5000, 64, 16))
    # pert = np.concatenate([np.concatenate([pertr,-perti], axis=2), np.concatenate([perti,pertr], axis=2)], axis=1)
    # train_data = train_data_ref + pert
    # test_data  = test_data_ref  + pert
    # Train:

    #    if it % 10000000 == 0:
    #        print "H is fixed"
    #        rnd_ = 50#np.random.randint(0, 0.8 * H_dataset.shape[0])
    # train_data = np.array([H_dataset[rnd_]])
    # test_data = np.array([H_dataset[rnd_]])

    feed_dict = {
        batch_size: args.batch_size,
        lr: args.learn_rate,
        snr_db_max: params['SNR_dB_max'],
        snr_db_min: params['SNR_dB_min'],
        # train_layer_no: tln_,
    }
    if args.data:
        # sample_ids = np.random.randint(0, np.shape(train_data)[0], params['batch_size'])
        # feed_dict[H] = train_data[sample_ids]
        feed_dict[H] = generate_channel(
            params['batch_size'], 2 * params['N'], 2 * params['K'], compl, rho, Phi_R12)
    if record_flag:
        feed_dict_test = {
            batch_size: args.test_batch_size,
            lr: args.learn_rate,
            snr_db_max: params['SNR_dB_max'],
            snr_db_min: params['SNR_dB_min'],
            # train_layer_no: tln_,
            # H: train_data[np.tile(sample_ids[0],(args.test_batch_size))],
        }
        if args.data:
            # sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
            # feed_dict[H] = test_data[sample_ids]
            feed_dict[H] = generate_channel(
                args.test_batch_size, 2 * params['N'], 2 * params['K'], compl, rho, Phi_R12)
        before_acc = 1.-sess.run(accuracy, feed_dict_test)
        record['before'].append(before_acc)

    # _, train_summary_ = sess.run([train, summary], feed_dict)
    sess.run(train, feed_dict)

    # Test
    if (it % args.test_every == 0):
        feed_dict = {
            batch_size: args.test_batch_size,
            snr_db_max: params['SNR_dB_max'],
            snr_db_min: params['SNR_dB_max'],
            # train_layer_no: tln_,
        }
        if args.data:
            # sample_ids = np.random.randint(0, np.shape(test_data)[0], args.test_batch_size)
            # feed_dict[H] = test_data[sample_ids]
            feed_dict[H] = generate_channel(
                args.test_batch_size, 2 * params['N'], 2 * params['K'], compl, rho, Phi_R12)
        if args.log:
            # test_accuracy_, test_loss_, logs_, x_, H_, test_summary_= sess.run([accuracy, loss, logs, x, H, summary], feed_dict)
            test_accuracy_, test_loss_, logs_, x_, H_ = sess.run(
                [accuracy, loss, logs, x, H], feed_dict)
            np.save('log.npy', logs_)
            break
        else:
            # if args.data:
            # result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'], mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
            # print "iteration", it, result
            # test_accuracy_, test_loss_, test_summary_, measured_snr_ = sess.run([accuracy, loss, summary, measured_snr], feed_dict)
            test_accuracy_, test_loss_, measured_snr_ = sess.run(
                [accuracy, loss, measured_snr], feed_dict)
            print((it, 'SER: {:.2E}'.format(
                1. - test_accuracy_), test_loss_, measured_snr_))
        if args.corr_analysis:
            log_ = sess.run(logs, feed_dict)
            for l in range(1, int(args.layers)+1):
                c = log_['layer'+str(l)]['linear']['I_WH']
                print((np.linalg.norm(c, axis=(1, 2))[0]))

            # temp2 = log_['layer'+str(l)]['linear']
            # np.save('W'+str(l)+'.npy', temp2['W'])
            # np.save('H'+str(l)+'.npy', temp2['H'])
            # np.save('WsetR'+str(l)+'.npy', temp2['WsetR'])
            # np.save('WsetI'+str(l)+'.npy', temp2['WsetI'])
            # print "Written"

            # np.save('Vw'+str(l)+'.npy', temp2['svd'][2][0])
            # np.save('Uh'+str(l)+'.npy', temp2['svd'][4][0])
            # np.save('Uw'+str(l)+'.npy', temp2['svd'][1][0])
            # np.save('WH'+str(l)+'.npy', temp2['WH'][0])
            # print np.matmul(temp2[1][0], temp2[4][0].conj().T)
            # print temp2['VhVwt'][0]
            # print temp2['norm_diff_U'][0], temp2['norm_Uw'][0], temp2['norm_diff_V'][0], temp2['norm_V'][0]
        # saver.save(sess, './reports/'+args.saveas, global_step=it)
        # test_summary_writer.add_summary(test_summary_, it)
        # train_summary_writer.add_summary(train_summary_, it)
test_data = generate_channel(
    args.test_batch_size, 2 * params['N'], 2 * params['K'], compl, rho, Phi_R12)
result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'],
                    mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
print(result)
# SNR_dBs = np.linspace(params['SNR_dB_min'],params['SNR_dB_max'],params['SNR_dB_max']-params['SNR_dB_min']+1)
# accs_mmse = np.zeros(shape=SNR_dBs.shape)
# accs_NN = np.zeros(shape=SNR_dBs.shape)
# iterations = 30
# SER Simulations
# print "PERTURBATIONS ON"
# if args.data:
#    test_data = test_data_ref
# for i in range(SNR_dBs.shape[0]):
#    noise_ = []
#    error_ = []
#    for j in range(iterations):
#        #pertr = np.random.normal(0.,0.01, (5000, 64, 16))
#        #perti = np.random.normal(0.,0.01, (5000, 64, 16))
#        #pert = np.concatenate([np.concatenate([pertr,-perti], axis=2), np.concatenate([perti,pertr], axis=2)], axis=1)
#        #test_data  = test_data_ref  + pert
#        feed_dict = {
#                batch_size: 5000,
#                snr_db_max: SNR_dBs[i],
#                snr_db_min: SNR_dBs[i],
#                train_layer_no: params['TL'],
#            }
#        if args.data:
#            sample_ids = np.random.randint(0, np.shape(test_data)[0], 5000)
#            feed_dict[H] = test_data[sample_ids]
#        acc = sess.run([mmse_accuracy, accuracy], feed_dict)
#        accs_mmse[i] += acc[0]/iterations
#        accs_NN[i] += acc[1]/iterations
#    print "SER_mmse: ", 1. - accs_mmse
#    print "SER_NN: ", 1. - accs_NN


# Own modifications ----------------------------------------------------
saver = tf.train.Saver()

if params['modulation'] == 'QAM_4':
    M = 4
    mod = 'QPSK'
elif params['modulation'] == 'QAM_16':
    M = 16
    mod = 'QAM16'
elif params['modulation'] == 'QAM_64':
    M = 64
    mod = 'QAM64'
else:
    M = 4
    mod = 'QPSK'
# if compl == 1:
#     mod = 'QPSK'
# else:
#     mod = 'BPSK'

Nt = params['K']
Nt2 = 2 * args.x_size
Nr2 = 2 * args.y_size
L = params['L']
snr_shift = 10 * np.log10(np.log2(M))
mmnet_shift = 10 * np.log10(Nr2 / Nt2)
ebn0db_low = int(params['SNR_dB_min']) - snr_shift + mmnet_shift
ebn0db_high = int(params['SNR_dB_max']) - snr_shift + mmnet_shift

# Save results into file: for quick comparison


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


filename = params['denoiser_name'] + '_' + mod + '_{}_{}_{}_snr{}_{}'.format(
    Nt2, Nr2, L, int(np.round(ebn0db_low)), int(np.round(ebn0db_high))) + fn_ext
ospath = ''
path = os.path.join('curves', mod, '{}x{}'.format(Nt2, Nr2))
path2 = os.path.join('models', mod, '{}x{}'.format(Nt2, Nr2))
path3 = os.path.join(path2, filename)


# Save model for import in own script: for detailed evaluation
pathfile = os.path.join(ospath, path3, filename)  # + '.ckpt'
check_path(pathfile, verbose=1)
save_path = saver.save(sess, pathfile)  # , global_step = train_iter


# SER Simulations

EbN0 = np.linspace(-6, 36, 21)
SNR_dBs = EbN0 + snr_shift
# SNR_dBs = np.linspace(params['SNR_dB_min'],params['SNR_dB_max'],params['SNR_dB_max']-params['SNR_dB_min']+1)
accs_mmse = np.zeros(shape=SNR_dBs.shape)
accs_NN = np.zeros(shape=SNR_dBs.shape)
# iterations = 30
N_test = 10000
print("PERTURBATIONS ON")
for i in range(SNR_dBs.shape[0]):
    noise_ = []
    error_ = []
    it_max = 0
    # Own code
    acc_error = [[], []]
    # Own code
    # int(1000 * 10 ** 6 / (N_test * Nt * np.log2(M / 2)))
    while (np.sum(acc_error[0]) < 1000) and (it_max < 1000):
        # for j in range(iterations):
        # pertr = np.random.normal(0.,0.01, (5000, 64, 16))
        # perti = np.random.normal(0.,0.01, (5000, 64, 16))
        # pert = np.concatenate([np.concatenate([pertr,-perti], axis=2), np.concatenate([perti,pertr], axis=2)], axis=1)
        # test_data  = test_data_ref  + pert
        feed_dict = {
            batch_size: N_test,
            snr_db_max: SNR_dBs[i],
            snr_db_min: SNR_dBs[i],
            # train_layer_no: params['TL'],
        }
        if args.data:
            # sample_ids = np.random.randint(0, np.shape(test_data)[0], 5000)
            # feed_dict[H] = test_data[sample_ids]
            feed_dict[H] = generate_channel(
                N_test, 2 * params['N'], 2 * params['K'], compl, rho, Phi_R12)
        acc = sess.run([mmse_accuracy, accuracy], feed_dict)
        accs_mmse[i] += acc[0]
        accs_NN[i] += acc[1]
        # Own code
        acc_error[0].append((1 - acc[1]) * N_test * Nt * np.log2(M))
        it_max = it_max + 1
        print('it: {}, error: {}'.format(it_max, np.sum(acc_error[0])))
    accs_mmse[i] = accs_mmse[i] / it_max
    accs_NN[i] = accs_NN[i] / it_max
    print('it: {}, SER_mmse: {}, SER_NN: {}'.format(
        i, 1. - accs_mmse[i], 1. - accs_NN[i]))
    # print("SER_mmse: ".format(1. - accs_mmse))
    # print("SER_NN: ".format(1. - accs_NN))

# Saving first results

ser = 1. - accs_NN


# Last save for quick evaluation
pathfile = os.path.join(ospath, path, 'SER_' + filename + '.npz')
check_path(pathfile, verbose=0)
np.savez(pathfile, ebn0=EbN0, ser=ser)
