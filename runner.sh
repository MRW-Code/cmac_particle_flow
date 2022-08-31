#!/bin/bash

### Testing the different models for sf = 10
#mkdir ./model_testing_leak
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_leak/resnet18_sf_10.txt
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_leak/convnext_tiny_sf_10.txt
#python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_leak/swinv2_base_window12to24_192to384_22kft1k_sf_10.txt
#mkdir ./model_testing_repeat_1_leak
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_1_leak/resnet18_sf_10.txt
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_1_leak/convnext_tiny_sf_10.txt
#python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_1_leak/swinv2_base_window12to24_192to384_22kft1k_sf_10.txt
#mkdir ./model_testing_repeat_2_leak
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_2_leak/resnet18_sf_10.txt
##python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_2_leak/convnext_tiny_sf_10.txt
#python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 1 2>&1 | tee ./model_testing_repeat_2_leak/swinv2_base_window12to24_192to384_22kft1k_sf_10.txt


### Testing the resnet split factors
#mkdir ./resnet_split_factor_leak
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_leak/split_factor_2.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_leak/split_factor_4.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_leak/split_factor_6.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_leak/split_factor_8.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize --gpu_idx 1 2>&1 | tee ./resnet_split_factor_leak/split_factor_10.txt
#mkdir ./resnet_split_factor_repeat1_leak
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_2.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_4.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_6.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_8.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat1_leak/split_factor_10.txt
#mkdir ./resnet_split_factor_repeat2_leak
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_2.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_4.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_6.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_8.txt
#python3 main.py -b 8 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize --gpu_idx 1 2>&1 | tee ./resnet_split_factor_repeat2_leak/split_factor_10.txt

#mkdir ./vit_out_new_leak
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak/split_factor_10.txt
#
#mkdir ./vit_out_new_leak_repeat_leak
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak/split_factor_10.txt
#
#mkdir ./vit_out_new_leak_repeat_leak_2
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 --gpu_idx 1 2>&1 | tee ./vit_out_new_leak_repeat_leak_2/split_factor_10.txt


## SWIN MODEL SF TESTING
mkdir ./swin_out_new_leak
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 1 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_1.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 2 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_2.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 3 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_3.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 4 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_4.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 5 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_5.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 6 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_6.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 7 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_7.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 8 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_8.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 9 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak/split_factor_9.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 0 2>&1 | tee .swin_out_new_leak/split_factor_10.txt

mkdir ./swin_out_new_leak_repeat_leak
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 1 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_1.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 2 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_2.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 3 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_3.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 4 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_4.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 5 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_5.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 6 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_6.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 7 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_7.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 8 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_8.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 9 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak/split_factor_9.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 0 2>&1 | tee .swin_out_new_leak_repeat_leak/split_factor_10.txt

mkdir ./swin_out_new_leak_repeat_leak_2
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 1 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_1.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 2 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_2.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 3 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_3.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 4 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_4.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 5 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_5.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 6 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_6.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 7 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_7.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 8 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_8.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 9 --gpu_idx 0 2>&1 | tee ./swin_out_new_leak_repeat_leak_2/split_factor_9.txt
python3 main.py -b 8 -g 4 -v kfold_ttv --from_scratch --make_figs -m swinv2_base_window12to24_192to384_22kft1k --split_factor 10 --gpu_idx 0 2>&1 | tee .swin_out_new_leak_repeat_leak_2/split_factor_10.txt