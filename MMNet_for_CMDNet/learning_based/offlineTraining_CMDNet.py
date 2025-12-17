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


# Introducing own functions
import utilities.my_mimo_channel as mymimo

def generate_channel(Nb, Nr, Nt, complex_system=0, rho=0, Phi_R12=0):
    '''Generates complex or real-valued [Nr]x[Nt] channel according to one ring correlation model
    Nb: Number of channel realizations
    Nr: Number of receive antennas
    Nt: Number of transmit antennas
    complex_system: Complex (1) / real (0)
    Phi_R12: correlation matrices
    '''
    if rho > 1:
        Hr = mymimo.generate_channel_onering(Nb, Nr, Nt, Phi_R12, compl=complex_system)
    else:
        Hr = mymimo.generate_channel(Nb, Nr, Nt, compl=complex_system, rho=rho)
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

# Newly added parser arguments
parser.add_argument('--angularspread',
                    type=float,
                    required=False,
                    default=0,
                    help='AngularSpread in One Ring massive MIMO model')
parser.add_argument('--cellsector',
                    type=float,
                    required=False,
                    default=120,
                    help='Cell sector in One Ring massive MIMO model')
parser.add_argument('--complex',
                    type=int,
                    required=False,
                    default=1,
                    help='Cell sector in One Ring massive MIMO model')
parser.add_argument('--filename_extension',
                    type=str,
                    required=False,
                    default='',
                    help='File ending specifying details of the simulation')
parser.add_argument('--save_directory',
                    type=str,
                    required=False,
                    default='',
                    help='Save directory')


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
    'denoiser_name': args.denoiser,
    'complex': args.complex,
    'angular_spread': int(args.angularspread),
    'cell_sector': int(args.cellsector),
    'filename_extension': args.filename_extension,
    'save_directory': args.save_directory,
}

# -- One Ring correlation matrices --------------------
if params['angular_spread'] > 1:
    if params['complex'] == 1:
        Phi_R12 = mymimo.mimo_OneRingModel(N=int(2 * params['N'] / 2), angularSpread=params['angular_spread'], cell_sector=params['cell_sector'], compl=params['complex'])
    else:
        Phi_R12 = mymimo.mimo_OneRingModel(N=2 * params['N'], angularSpread=params['angular_spread'], cell_sector=params['cell_sector'], compl=params['complex'])
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
            params['batch_size'], 2 * params['N'], 2 * params['K'], params['complex'], params['angular_spread'], Phi_R12)
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
                args.test_batch_size, 2 * params['N'], 2 * params['K'], params['complex'], params['angular_spread'], Phi_R12)
        before_acc = 1.-sess.run(accuracy, feed_dict_test)
        record['before'].append(before_acc)

    # _, train_summary_ = sess.run([train, summary], feed_dict)
    sess.run(train, feed_dict)

    # Test
    if (it % args.test_every == 0) or (it == args.train_iterations - 1):
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
                args.test_batch_size, 2 * params['N'], 2 * params['K'], params['complex'], params['angular_spread'], Phi_R12)
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
print('Training completed.')
# Not useful? ------------------
# test_data = generate_channel(
#     args.test_batch_size, 2 * params['N'], 2 * params['K'], params['complex'], params['angular_spread'], Phi_R12)
# result = model_eval(test_data, params['SNR_dB_min'], params['SNR_dB_max'],
#                     mmse_accuracy, accuracy, batch_size, snr_db_min, snr_db_max, H, sess)
# print(result)
# ------------------------------
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
# if params['complex'] == 1:
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
    Nt2, Nr2, L, int(np.round(ebn0db_low)), int(np.round(ebn0db_high))) + params['filename_extension']




# Save model for import in own script: for detailed evaluation
path_curves = os.path.join('curves', mod, '{}x{}'.format(Nt2, Nr2))
path_models = os.path.join('models', mod, '{}x{}'.format(Nt2, Nr2), filename)
pathfile = os.path.join(params['save_directory'], path_models, filename)  # + '.ckpt'
check_path(pathfile, verbose=1)
save_path = saver.save(sess, pathfile)  # , global_step = train_iter


# SER Simulations

EbN0 = np.linspace(-6, 36, 21)
SNR_dBs = EbN0 + snr_shift
# SNR_dBs = np.linspace(params['SNR_dB_min'],params['SNR_dB_max'],params['SNR_dB_max']-params['SNR_dB_min']+1)
accs_mmse = np.zeros(shape=SNR_dBs.shape)
accs_NN = np.zeros(shape=SNR_dBs.shape)
# iterations = 30
N_test = args.test_batch_size # 10000
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
                N_test, 2 * params['N'], 2 * params['K'], params['complex'], params['angular_spread'], Phi_R12)
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
pathfile = os.path.join(params['save_directory'], path_curves, 'SER_' + filename + '.npz')
check_path(pathfile, verbose=0)
np.savez(pathfile, ebn0=EbN0, ser=ser)
