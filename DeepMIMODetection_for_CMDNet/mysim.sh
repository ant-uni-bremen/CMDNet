#!/bin/bash
source ~/.bashrc
conda activate detnet
nohup python2 DetNet_v2.py > DNet2014.out 2> DNet2014.err