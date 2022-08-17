#!/bin/bash
#mkdir ./batch_size_tests
#python3 main.py --gpu_idx 3 -b 16 --from_scratch 2>&1 | tee ./batch_size_tests/bs_16_output.txt
#python3 main.py --gpu_idx 3 -b 32 --from_scratch 2>&1 | tee ./batch_size_tests/bs_32_output.txt
#python3 main.py --gpu_idx 3 -b 64 --from_scratch 2>&1 | tee ./batch_size_tests/bs_64_output.txt
#python3 main.py --gpu_idx 3 -b 128 --from_scratch 2>&1 | tee ./batch_size_tests/bs_128_output.txt

mkdir ./vit_out
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 2>&1 | tee ./vit_out/split_factor_1.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 2>&1 | tee ./vit_out/split_factor_2.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 2>&1 | tee ./vit_out/split_factor_3.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 2>&1 | tee ./vit_out/split_factor_4.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 2>&1 | tee ./vit_out/split_factor_5.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 2>&1 | tee ./vit_out/split_factor_6.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 2>&1 | tee ./vit_out/split_factor_7.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 2>&1 | tee ./vit_out/split_factor_8.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 2>&1 | tee ./vit_out/split_factor_9.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 2>&1 | tee ./vit_out/split_factor_10.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 15 2>&1 | tee ./vit_out/split_factor_15.txt





