#!/bin/bash
#mkdir ./batch_size_tests
#python3 main.py --gpu_idx 3 -b 16 --from_scratch 2>&1 | tee ./batch_size_tests/bs_16_output.txt
#python3 main.py --gpu_idx 3 -b 32 --from_scratch 2>&1 | tee ./batch_size_tests/bs_32_output.txt
#python3 main.py --gpu_idx 3 -b 64 --from_scratch 2>&1 | tee ./batch_size_tests/bs_64_output.txt
#python3 main.py --gpu_idx 3 -b 128 --from_scratch 2>&1 | tee ./batch_size_tests/bs_128_output.txt

#mkdir ./vit_out_new
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 2>&1 | tee ./vit_out_new/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 2>&1 | tee ./vit_out_new/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 2>&1 | tee ./vit_out_new/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 2>&1 | tee ./vit_out_new/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 2>&1 | tee ./vit_out_new/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 2>&1 | tee ./vit_out_new/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 2>&1 | tee ./vit_out_new/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 2>&1 | tee ./vit_out_new/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 2>&1 | tee ./vit_out_new/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 2>&1 | tee ./vit_out_new/split_factor_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 15 2>&1 | tee ./vit_out_new/split_factor_15.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 20 2>&1 | tee ./vit_out_new/split_factor_20.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 25 2>&1 | tee ./vit_out_new/split_factor_25.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 30 2>&1 | tee ./vit_out_new/split_factor_30.txt

#mkdir ./vit_out_new_repeat
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 2>&1 | tee ./vit_out_new_repeat/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 2>&1 | tee ./vit_out_new_repeat/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 2>&1 | tee ./vit_out_new_repeat/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 2>&1 | tee ./vit_out_new_repeat/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 2>&1 | tee ./vit_out_new_repeat/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 2>&1 | tee ./vit_out_new_repeat/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 2>&1 | tee ./vit_out_new_repeat/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 2>&1 | tee ./vit_out_new_repeat/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 2>&1 | tee ./vit_out_new_repeat/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 2>&1 | tee ./vit_out_new_repeat/split_factor_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 15 2>&1 | tee ./vit_out_new_repeat/split_factor_15.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 20 2>&1 | tee ./vit_out_new_repeat/split_factor_20.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 25 2>&1 | tee ./vit_out_new_repeat/split_factor_25.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 30 2>&1 | tee ./vit_out_new_repeat/split_factor_30.txt


## To Run
#mkdir ./vit_out_new_no_augs
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 1 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_1.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 2 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_2.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 3 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_3.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 4 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_4.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 5 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_5.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 6 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_6.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 7 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 8 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 9 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 10 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 15 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_15.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 20 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_20.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 25 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_25.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m vit_tiny_patch16_384 --split_factor 30 --no_augs 2>&1 | tee ./vit_out_new_no_augs/split_factor_30.txt

#mkdir ./model_testing
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 7 2>&1 | tee ./model_testing/resnet18_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 7 2>&1 | tee ./model_testing/convnext_tiny_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 8 2>&1 | tee ./model_testing/resnet18_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 8 2>&1 | tee ./model_testing/convnext_tiny_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 9 2>&1 | tee ./model_testing/resnet18_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 9 2>&1 | tee ./model_testing/convnext_tiny_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing/resnet18_sf_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing/convnext_tiny_sf_10.txt
#
#mkdir ./model_testing_repeat_1
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 7 2>&1 | tee ./model_testing_repeat_1/resnet18_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 7 2>&1 | tee ./model_testing_repeat_1/convnext_tiny_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 8 2>&1 | tee ./model_testing_repeat_1/resnet18_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 8 2>&1 | tee ./model_testing_repeat_1/convnext_tiny_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 9 2>&1 | tee ./model_testing_repeat_1/resnet18_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 9 2>&1 | tee ./model_testing_repeat_1/convnext_tiny_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing_repeat_1/resnet18_sf_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing_repeat_1/convnext_tiny_sf_10.txt
#
#mkdir ./model_testing_repeat_2
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 7 2>&1 | tee ./model_testing_repeat_2/resnet18_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 7 2>&1 | tee ./model_testing_repeat_2/convnext_tiny_sf_7.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 8 2>&1 | tee ./model_testing_repeat_2/resnet18_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 8 2>&1 | tee ./model_testing_repeat_2/convnext_tiny_sf_8.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 9 2>&1 | tee ./model_testing_repeat_2/resnet18_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 9 2>&1 | tee ./model_testing_repeat_2/convnext_tiny_sf_9.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m resnet18 --split_factor 10 2>&1 | tee ./model_testing_repeat_2/resnet18_sf_10.txt
#python3 main.py -b 32 -v kfold_ttv --from_scratch --make_figs -m convnext_tiny --split_factor 10 2>&1 | tee ./model_testing_repeat_2/convnext_tiny_sf_10.txt

mkdir ./resnet_split_factor
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 2 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_2.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 4 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_4.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 6 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_6.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 8 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_8.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 10 --no_resize 2>&1 | tee ./resnet_split_factor/split_factor_10.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 1 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_1.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 3 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_3.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 5 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_5.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 7 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_7.txt
python3 main.py -b 32 -v kfold_ttv --from_scratch -m resnet18 --split_factor 9 --no_resize  2>&1 | tee ./resnet_split_factor/split_factor_9.txt
