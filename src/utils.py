import argparse
import os

parser = argparse.ArgumentParser(usage='python main.py')

parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
parser.add_argument('-s', '--split_factor', action='store', dest='split_factor', type=int,
                    default=3, choices=range(0, 60))
parser.add_argument('-b', '--batch_size', action='store', dest='batch_size', type=int,
                    default=8, choices=range(0, 1000))
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)
parser.add_argument('--make_figs', action='store_true', dest='make_figs', default=False)
parser.add_argument('--from_scratch', action='store_true', dest='from_scratch')
parser.add_argument('-v', '--cv_method', action='store', dest='cv_method', default='split_first',
                  choices=['kfold', 'crop_fold', 'split_first', 'ttv', 'kfold_ttv'])
parser.add_argument('-g', '--grad_accum', action='store', dest='grad_accum', type=int,
                    default=1, choices=range(0, 50))
parser.add_argument('-m', '--model', action='store', dest='model', default='resnet18',
                  choices=['convnext_tiny_in22k', 'resnet18', 'convnext_tiny',
                           'convnext_small', 'resnet50', 'vit_tiny_patch16_384',
                           'swinv2_tiny_window16_256', 'swinv2_cr_tiny_384',
                           'vit_small_patch16_384', 'vit_small_patch32_384', 'swinv2_cr_tiny_384',
                           'swinv2_base_window12to24_192to384_22kft1k'])
parser.add_argument('--no_resize', action='store_true', dest='no_resize')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
os.environ['HOME'] = './temp_home'  ##### REMOVE AFTER HEX MENDED
