from src.utils import args
import os
from pathlib import Path
import glob
import shutil

def paths_from_dir(dir_path):
    dir_path = Path(dir_path)
    img_paths = []
    for paths in dir_path.iterdir():
        for path in paths.iterdir():
            img_paths.append(path)
    return img_paths

def make_needed_dirs():
    os.makedirs('./aug_images/Cohesive', exist_ok=True)
    os.makedirs('./aug_images/Easyflowing', exist_ok=True)
    os.makedirs('./aug_images/Freeflowing', exist_ok=True)
    os.makedirs('./checkpoints/splitting_test/models/{args.model}', exist_ok=True)
    os.makedirs('./checkpoints/splitting_test/figures', exist_ok=True)
    os.makedirs('./split_images/valid/Cohesive', exist_ok=True)
    os.makedirs('./split_images/valid/Easyflowing', exist_ok=True)
    os.makedirs('./split_images/valid/Freeflowing', exist_ok=True)
    os.makedirs('./split_images/train/Cohesive', exist_ok=True)
    os.makedirs('./split_images/train/Easyflowing', exist_ok=True)
    os.makedirs('./split_images/train/Freeflowing', exist_ok=True)
    os.makedirs('./preds_csv', exist_ok=True)

def delete_file(path):
    shutil.rmtree(path)