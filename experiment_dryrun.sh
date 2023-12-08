#!/bin/sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train_eval.py --data_path "dry_run_data/ASL" --num_workers 4 --batchsize 2 --lr 0.01 --epochs 10 --name "testrun"