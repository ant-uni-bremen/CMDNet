#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 13 14:31:11 2020

@author: beck
"""
# Own packages
import utilities.my_functions as mf
import utilities.my_communications as com
import utilities.my_training as mt
# Include parent folder
import sys
sys.path.append('..')


# Convert own save files into other file format

mod = com.modulation('QPSK')  # e.g., BPSK, QPSK, QAM16, QAM64, ASK4, ASK8
sim_params = mf.simulation_parameters(16, 16, 16, mod, 10000, [-6, 36])
# Filename
sel_algo = 'AMP'  # 'MMSE', 'AMP', 'CMD', 'CMD_fixed'
# e.g., _binary, _hoydis, _snr13_36
fn = mf.filename_module(sim_params, 'curves',
                        'trainhist_', sel_algo, '_binary')
saveobj = mf.savemodule()

form_from = 'json'
form_to = 'npz'


# Load file
train_hist = mt.TrainingHistory()
train_hist.filename = fn.pathfile
json_flag = form_from == 'json'
train_hist.dict2obj(saveobj.load(train_hist.filename,
                    form=form_from), json=json_flag)


perf_meas = mf.performance_measures()
fn.typename = 'RES_'
fn.generate_pathfile_MIMO()
perf_meas.load_results(saveobj.load(fn.pathfile, form=form_from))

# Save into other format

if train_hist.epoch != []:
    saveobj.save(train_hist.filename, train_hist.obj2dict(),
                 form=form_to, verbose=1)

saveobj.save(fn.pathfile, perf_meas.results(), form=form_to, verbose=1)
