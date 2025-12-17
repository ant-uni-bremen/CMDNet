#!/bin/bash
source ~/.bashrc
conda activate ml1
python -u offlineTraining_CMDNet.py --x-size 32 --y-size 32 --snr-min 7 --snr-max 30 --layers 64 -lr 0.001 --batch-size 500 --train-iterations 200 --mod QAM_4  --test-batch-size 10000 --linear MMNet_iid  --denoiser MMNet  --test-every 100 --gpu 1 --complex 1 --angularspread 0 --cellsector 120 --filename_extension _test --save_directory /home/beck/Nextcloud/Promotion/Python/ML_Transceiver/CMDNet > MMNet_test.out 2> MMNet_test.err