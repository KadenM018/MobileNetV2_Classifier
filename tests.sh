#!/bin/bash
# List of arguments for each experiment
experiments=(
    "--data_path data/ASL_Test --weights_dir saves/exp05/weights/best_weights.pth --save_dir saves/test_results --num_workers 4 --name test_exp05"
    # Add more experiments as needed
)

# Loop through each experiment
for args in "${experiments[@]}"; do
    echo "Running experiment with arguments: $args"
    # You can preload tcmalloc here if needed
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4 python test_alt.py $args && echo "Test successful" || echo "Test failed"
done

echo "All tests completed"
