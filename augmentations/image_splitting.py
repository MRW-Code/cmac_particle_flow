import cv2
from pathlib import Path
import tqdm
import numpy as np
class ImageSplitter():
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
                    labels_cropped.append(label)
                    self.all_crops.append(self.cropped_image[i])

        self.raw_test = list(zip(self.all_raw, labels_raw))
        self.raw_train = list(zip(self.all_crops, labels_cropped))


    # def cropping(self, num_quadrants):
    #     '''
    #     splits the big images into 4 quardants
    #     :param num_quadrants: has to be 4 unless code updated - but easy update
    #     :return: 4 smaller images of each corner of the original image
    #     '''
    #     # self.cropped_images = []
    #     x = self.raw.shape[0]
    #     y = self.raw.shape[1]
    #     if num_quadrants == 4:
    #         # crop1 = self.raw[0:int(x/2), 0:int(y/2)]
    #         # crop2 = self.raw[int(x/2):x, 0:int(y/2)]
    #         # crop3 = self.raw[0:int(x/2), int(y/2):y]
    #         # crop4 = self.raw[int(x/2):x,  int(y/2):y]
    #         crop1 = self.raw[0:1745, 0:2680]
    #         crop2 = self.raw[1746:x, 0:2680]
    #         crop3 = self.raw[0:1745, 2679:y]
    #         crop4 = self.raw[1746:x, 2679:y]
    #     else:
    #         raise AttributeError('Code only works for 4 quadrants - amend if needed')
    #     return crop1, crop2, crop3, crop4

    # def pick_split(self):
    #     '''
    #     a pretty dirty function to shuffle which of the images ends up in the test/train splits
    #     honestly not sure it is gonna do much as you have to recall all the augmentations
    #     defo a better way to do this but i guess it works for now
    #     :return: a "raw" image for val set and 3 also raw technically images for train set
    #     the 3 will get augmented later.
    #     '''
    #     self.cropped_image = []
    #     if self.split_number == 0:
    #         self.unchanged, crop1, crop2, crop3 = self.cropping(4)
    #     elif self.split_number == 1:
    #         crop3, self.unchanged, crop1, crop2 = self.cropping(4)
    #     elif self.split_number == 2:
    #         crop2, crop3, self.unchanged, crop1 = self.cropping(4)
    #     elif self.split_number == 3:
    #         crop1, crop2, crop3, self.unchanged = self.cropping(4)
    #     else:
    #         raise AttributeError('Split number out of range, expected range 0-3')
    #     self.cropped_image.append(crop1)
    #     self.cropped_image.append(crop2)
    #     self.cropped_image.append(crop3)

    # def do_splitting(self):
    #     '''
    #     calls the cropping function with the split allocation.
    #     :return: zip of images and labels, seperate test/train sets
    #     '''
    #     self.all_raw = []
    #     self.all_crops = []
    #     labels_raw = []
    #     labels_cropped = []
    #     for paths in self.data_path.iterdir():
    #         for path in tqdm.tqdm(paths.iterdir()):
    #             label = path.parent.name
    #             self.open_img(str(path))
    #             self.cropping()
    #             self.pick_split()
    #             # print(self.cropped_image)
    #             # print(self.raw)
    #             self.all_raw.append(self.unchanged)
    #             labels_raw.append(label)
    #             for i in range(3):
    #                 labels_cropped.append(label)
    #                 self.all_crops.append(self.cropped_image[i])
    #
    #     self.raw_test = list(zip(self.all_raw, labels_raw))
    #     self.raw_train = list(zip(self.all_crops, labels_cropped))
