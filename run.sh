#!/bin/bash
python3 main.py --gpu_idx 3 -s 1 2>&1 | tee ./split_factor_tests/split_factor_1.txt
python3 main.py --gpu_idx 3 -s 2 2>&1 | tee ./split_factor_tests/split_factor_2.txt
python3 main.py --gpu_idx 3 -s 3 2>&1 | tee ./split_factor_tests/split_factor_3.txt
python3 main.py --gpu_idx 3 -s 4 2>&1 | tee ./split_factor_tests/split_factor_4.txt
python3 main.py --gpu_idx 3 -s 5 2>&1 | tee ./split_factor_tests/split_factor_5.txt
python3 main.py --gpu_idx 3 -s 6 2>&1 | tee ./split_factor_tests/split_factor_6.txt
python3 main.py --gpu_idx 3 -s 7 2>&1 | tee ./split_factor_tests/split_factor_7.txt
python3 main.py --gpu_idx 3 -s 8 2>&1 | tee ./split_factor_tests/split_factor_8.txt
python3 main.py --gpu_idx 3 -s 9 2>&1 | tee ./split_factor_tests/split_factor_9.txt
python3 main.py --gpu_idx 3 -s 10 2>&1 | tee ./split_factor_tests/split_factor_10.txt


