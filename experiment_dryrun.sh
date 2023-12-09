#!/bin/sh
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train_eval.py --data_path "run400/ASL" --bestonly --num_workers 4 --batchsize 20 --lr 0.01 --epochs 4 --name "testrun4"