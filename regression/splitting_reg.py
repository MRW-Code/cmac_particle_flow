import cv2
from pathlib import Path
import tqdm
import numpy as np
from multiprocessing import Process
import pandas as pd

class RegressionImageSplitter():
    '''
    splits the images where split number is the index of the point to take for train/test split
    split factor is the number of splits along the axis taken - how many sub images there will be
    number of images = split factor^2
    '''

    def __init__(self, path_to_data, split_number, split_factor):
        self.raw = None
        self.data_path = Path(path_to_data)
        self.unchanged = None
        self.cropped_image = None
        self.split_number = split_number
        self.all_raw = None
        self.all_crops = None
        self.raw_test = None
        self.raw_train = None
        self.split_factor = split_factor

    def open_img(self, img_path):
        self.raw = cv2.imread(img_path)

    def cropping(self):
        '''
        splits the big images into 4 quardants
        :param split_factor: has to be 4 unless code updated - but easy update
        :return: 4 smaller images of each corner of the original image
        '''
        # self.cropped_images = []
        split_factor = self.split_factor + 1
        x = self.raw.shape[0]
        y = self.raw.shape[1]
        points_x = np.linspace(0, x, split_factor, dtype=np.int)
        points_y = np.linspace(0, y, split_factor, dtype=np.int)
        crops = []
        for i in range(len(points_x) - 1):
            for j in range(len(points_y) - 1):
                crops.append(self.raw[points_x[i]:points_x[i] + points_x[1], points_y[j]:points_y[j] + points_y[1]])
        return crops

    def pick_split(self, crops):
        self.unchanged = crops[self.split_number]
        crops.pop(self.split_number)
        self.cropped_image = crops


    def do_splitting(self):
        '''
        calls the cropping function with the split allocation.
        :return: zip of images and labels, seperate test/train sets
        '''
        self.all_raw = []
        self.all_crops = []
        labels_raw = []
        labels_cropped = []
        api_raw = []
        api_crops = []
        for paths in self.data_path.iterdir():
            for path in tqdm.tqdm(paths.iterdir()):
                api = path.parts[3][0:-4]
                label = path.parent.name
                self.open_img(str(path))
                crops = self.cropping()
                self.pick_split(crops)
                # print(self.cropped_image)
                # print(self.raw)
                self.all_raw.append(self.unchanged)
                labels_raw.append(label)
                api_raw.append(api)
                for i in range(len(self.cropped_image)):
                    labels_cropped.append(label)
                    self.all_crops.append(self.cropped_image[i])
                    api_crops.append(path.parts[3][0:-4])
        self.raw_test = list(zip(self.all_raw, labels_raw, api_raw))
        self.raw_train = list(zip(self.all_crops, labels_cropped, api_crops))


    def do_oversampled_splitting(self, num_free, num_co, num_ez):
        '''
        calls the cropping function with the split allocation and oversampling .
        :return: zip of images and labels, seperate test/train sets
        '''
        self.all_raw = []
        self.all_crops = []
        labels_raw = []
        labels_cropped = []
        for paths in self.data_path.iterdir():
            for path in tqdm.tqdm(paths.iterdir()):
                label = path.parent.name
                self.open_img(str(path))
                crops = self.cropping()
                self.pick_split(crops)
                # print(self.cropped_image)
                # print(self.raw)
                self.all_raw.append(self.unchanged)
                labels_raw.append(label)
                for i in range(len(self.cropped_image)):
                    if label == 'cohesive':
                        for x in range(num_co):
                            labels_cropped.append(label)
                            self.all_crops.append(self.cropped_image[i])
                    elif label == 'freeflowing':
                        for y in range(num_free):
                            labels_cropped.append(label)
                            self.all_crops.append(self.cropped_image[i])
                    elif label == 'easyflowing':
                        for z in range(num_ez):
                            labels_cropped.append(label)
                            self.all_crops.append(self.cropped_image[i])
                    else:
                        AttributeError('Unexpected class name')

        print(pd.Series(labels_cropped).value_counts())
        self.raw_test = list(zip(self.all_raw, labels_raw))
        self.raw_train = list(zip(self.all_crops, labels_cropped))
        print('done')
