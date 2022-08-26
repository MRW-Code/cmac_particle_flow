#!/bin/bash

## Testing the different models for sf = 10
mkdir ./model_testing_leak
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing_leak/resnet18_sf_10.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing_leak/convnext_tiny_sf_10.txt
mkdir ./model_testing_repeat_1_leak
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing_repeat_1_leak/resnet18_sf_10.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing_repeat_1_leak/convnext_tiny_sf_10.txt
mkdir ./model_testing_repeat_2_leak
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing_repeat_2_leak/resnet18_sf_10.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing_repeat_2_leak/convnext_tiny_sf_10.txt

## Testing the resnet split factors
mkdir ./resnet_split_factor_leak
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  2>&1 | tee ./resnet_split_factor_leak/split_factor_2.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  2>&1 | tee ./resnet_split_factor_leak/split_factor_4.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  2>&1 | tee ./resnet_split_factor_leak/split_factor_6.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  2>&1 | tee ./resnet_split_factor_leak/split_factor_8.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize 2>&1 | tee ./resnet_split_factor_leak/split_factor_10.txt
mkdir ./resnet_split_factor_repeat1_leak
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_2.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_4.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_6.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_8.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_10.txt
mkdir ./resnet_split_factor_repeat2_leak
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_2.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_4.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_6.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_8.txt
python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_10.txt


