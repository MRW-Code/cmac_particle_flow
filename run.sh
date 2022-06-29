#!/bin/bash
mkdir ./batch_size_tests
python3 main.py --gpu_idx 3 -b 16 --from_scratch 2>&1 | tee ./batch_size_tests/bs_16_output.txt
python3 main.py --gpu_idx 3 -b 32 --from_scratch 2>&1 | tee ./batch_size_tests/bs_32_output.txt
python3 main.py --gpu_idx 3 -b 64 --from_scratch 2>&1 | tee ./batch_size_tests/bs_64_output.txt
python3 main.py --gpu_idx 3 -b 128 --from_scratch 2>&1 | tee ./batch_size_tests/bs_128_output.txt





