import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL.Image import fromarray
from src.helpers import delete_file

class ImageSplitter():

    def __init__(self, img_paths, split_factor, val_idx):
        self.img_paths = img_paths
        self.split_factor = split_factor
        self.val_idx = val_idx

    def open_img(self, path): return cv2.imread(str(path))

    def split_images(self, img_pth):
        crops = []
        img = self.open_img(img_pth)
        x = img.shape[0]
        y = img.shape[1]
        if self.split_factor == 0:
            crops.append(img)
        elif self.split_factor == 1:
            crops.append(img[0:(x // 2)])
            crops.append(img[(x // 2):-1])
        else:
            points_x = np.linspace(0, x, self.split_factor + 1, dtype=np.int)
            points_y = np.linspace(0, y, self.split_factor + 1, dtype=np.int)
            for i in range(len(points_x) - 1):
                for j in range(len(points_y) - 1):
                    crops.append(img[points_x[i]:points_x[i] + points_x[1], points_y[j]:points_y[j] + points_y[1]])
        return crops

    def do_splitting(self):
        print(f'Cropping images with idx = {self.val_idx} as test set')
        train_set, train_names, train_labels = [], [], []
        val_set, val_names, val_labels = [], [], []
        for img in tqdm(self.img_paths, nrows=80):
            img_crops = self.split_images(img)
            label = img.parent.name
            if self.val_idx is not None and self.split_factor != 0:
                for idx, crop in enumerate(img_crops):
                    if idx == self.val_idx:
                        val_set.append(crop)
                        val_labels.append(label)
                        val_names.append(f'{idx}_{img.name}')
                    else:
                        train_set.append(crop)
                        train_labels.append(label)
                        train_names.append(f'{idx}_{img.name}')
            else:
                for idx, crop in enumerate(img_crops):
                    train_set.append(crop)
                    train_labels.append(label)
                    train_names.append(f'{idx}_{img.name}')
        training_data = list(zip(train_set, train_names, train_labels))
        validation_data = list(zip(val_set, val_names, val_labels)) if len(val_set) > 1 else None
        return training_data, validation_data

    def save_split_images(self):
        training_data, validation_data = self.do_splitting()
        for img, name, label in training_data:
            image = fromarray(img)
            image.save(f'./split_images/train/{label}/{name}')
        if validation_data is not None:
            for img, name, label in validation_data:
                image = fromarray(img)
                image.save(f'./split_images/valid/{label}/{name}')

    def save_split_first(self, X_train, X_val):
        print('SAVING SPLIT IMAGES TRAIN')
        delete_file('./split_images/train/')
        self.img_paths = X_train
        training_data, _ = self.do_splitting()
        for img, name, label in tqdm(training_data, nrows=80):
            os.makedirs(f'./split_images/train/{label}', exist_ok=True)
            image = fromarray(img)
            image.save(f'./split_images/train/{label}/{name}')

        print('SAVING SPLIT IMAGES TEST')
        delete_file('./split_images/valid/')
        self.img_paths = X_val
        validation_data, _ = self.do_splitting()
        for img, name, label in validation_data:
            os.makedirs(f'./split_images/valid/{label}', exist_ok=True)
            image = fromarray(img)
            image.save(f'./split_images/valid/{label}/{name}')
