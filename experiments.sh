#!/bin/sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train_eval.py --data_path "run8k/ASL" --bestonly --num_workers 4 --batchsize 64 --lr 0.02 --epochs 10 --name "run_10epoch_lr02_batch64"