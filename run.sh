#!/bin/bash
python3 main.py --gpu_idx 3 -s 1 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_1.txt
python3 main.py --gpu_idx 3 -s 2 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_2.txt
python3 main.py --gpu_idx 3 -s 3 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_3.txt
python3 main.py --gpu_idx 3 -s 4 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_4.txt
python3 main.py --gpu_idx 3 -s 5 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_5.txt
python3 main.py --gpu_idx 3 -s 6 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_6.txt
python3 main.py --gpu_idx 3 -s 7 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_7.txt
python3 main.py --gpu_idx 3 -s 8 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_8.txt
python3 main.py --gpu_idx 3 -s 9 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_9.txt
python3 main.py --gpu_idx 3 -s 10 --from_scratch 2>&1 | tee ./split_factor_tests/split_factor_10.txt


