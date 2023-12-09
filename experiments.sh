#!/bin/bash
# List of arguments for each experiment
experiments=(
    #"--data_path run400/ASL --bestonly --num_workers 4 --batchsize 20 --lr 0.005 --epochs 1 --name testscriptrun"
    #"--data_path run400/ASL --bestonly --num_workers 4 --batchsize 20 --lr 0.005 --epochs 1 --name testscriptrun2"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 64 --lr 0.0008 --epochs 5 --name exp01"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 64 --lr 0.002 --epochs 5 --name exp02"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 64 --lr 0.005 --epochs 5 --name exp03"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 64 --lr 0.01 --epochs 5 --name exp04"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 32 --lr 0.0008 --epochs 5 --name exp05"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 32 --lr 0.002 --epochs 5 --name exp06"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 32 --lr 0.005 --epochs 5 --name exp07"
    "--data_path run8k/ASL --bestonly --num_workers 4 --batchsize 32 --lr 0.01 --epochs 5 --name exp08"
    # Add more experiments as needed
)

# Loop through each experiment
for args in "${experiments[@]}"; do
    echo "Running experiment with arguments: $args"
    # You can preload tcmalloc here if needed
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python train_eval.py $args && echo "Experiment successful" || echo "Experiment failed"
done

echo "All experiments completed"
