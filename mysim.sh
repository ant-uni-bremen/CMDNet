#!/bin/bash
source ~/.bashrc
conda activate machine_learning
nohup python -u learning_based/offlineTraining_costum.py --x-size 32 --y-size 64 --snr-min 4 --snr-max 27 --layers 64 -lr 0.001 --batch-size 500 --train-iterations 100000 --mod QAM_4  --test-batch-size 10000 --linear MMNet_iid  --denoiser MMNet  --test-every 500 > MN64128OR.out 2> MN64128OR.err