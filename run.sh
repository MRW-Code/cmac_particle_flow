#!/bin/bash
python3 main.py --gpu_idx 3 -s 1 2>&1 | tee ./split_factor_tests/split_factor_1.txt
python3 main.py --gpu_idx 2 -s 1 2>&1 | tee ./split_factor_tests/split_factor_2.txt
python3 main.py --gpu_idx 3 -s 1 2>&1 | tee ./split_factor_tests/split_factor_3.txt
python3 main.py --gpu_idx 4 -s 1 2>&1 | tee ./split_factor_tests/split_factor_4.txt
python3 main.py --gpu_idx 5 -s 1 2>&1 | tee ./split_factor_tests/split_factor_5.txt
python3 main.py --gpu_idx 6 -s 1 2>&1 | tee ./split_factor_tests/split_factor_6.txt
python3 main.py --gpu_idx 7 -s 1 2>&1 | tee ./split_factor_tests/split_factor_7.txt
python3 main.py --gpu_idx 8 -s 1 2>&1 | tee ./split_factor_tests/split_factor_8.txt
python3 main.py --gpu_idx 9 -s 1 2>&1 | tee ./split_factor_tests/split_factor_9.txt
python3 main.py --gpu_idx 10 -s 1 2>&1 | tee ./split_factor_tests/split_factor_10.txt


