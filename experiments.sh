#!/bin/sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train_eval.py --data_path "dry_run_data/ASL" --bestonly --num_workers 4 --batchsize 20 --lr 0.01 --epochs 10 --name "testrun3"