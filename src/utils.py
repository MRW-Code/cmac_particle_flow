import argparse
import os

parser = argparse.ArgumentParser(usage='python main.py')

parser.add_argument('--gpu_idx', action='store', dest='gpu_idx', default='0',
                  choices=['0', '1', '2', '3', '4', '5'])
parser.add_argument('-s', '--split_factor', action='store', dest='split_factor', type=int,
                    default=2, choices=range(0, 15))
parser.add_argument('--no_augs', action='store_true', dest='no_augs', default=False)
parser.add_argument('--from_scratch', action='store_true', dest='from_scratch')
parser.add_argument('-v', '--cv_method', action='store', dest='cv_method', default='split_first',
                  choices=['kfold', 'crop_fold', 'split_first'])

args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx