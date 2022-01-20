from splitting_reg import RegressionImageSplitter
import cv2
import tqdm
import gc
import numpy as np
from PIL.Image import fromarray
import concurrent.futures
import multiprocessing
from multiprocessing import Process
from itertools import repeat
import threading

class RegressionImageAugmentor(RegressionImageSplitter):
    '''
    class does the augmenting on the already split images.
    '''

    def __init__(self, path_to_data, split_number, split_factor, save_augs, save_raw, oversample):
        RegressionImageSplitter.__init__(self,path_to_data, split_number, split_factor)

        self.oversample = oversample
        if self.oversample == True:
            self.do_oversampled_splitting(num_free=1, num_co=9, num_ez=3)
        else:
            self.do_splitting()

        self.memory_cleaner()
        self.save_augs = save_augs
        self.img = None
        self.flipped_images = None
        self.rotated_images = None
        self.aug_imgs = None
        self.save_raw = save_raw

    def memory_cleaner(self):
        '''
        ashamed of this... needed the ram
        :return:
        '''
        del self.all_crops
        del self.all_raw
        del self.unchanged
        del self.cropped_image
        del self.split_number
        del self.raw
        gc.collect()


    def rotating(self, num_rotations):
        '''
        rotates the images through increments
        :param num_rotations:
        :return: rotated images list
        '''
        (h, w) = self.img.shape[:2]
        center = (w / 2, h / 2)
        self.rotated_images = []
        for i in range(num_rotations):
            rotations = list(range(0, 180, num_rotations))
            M = cv2.getRotationMatrix2D(center, rotations[i], 1.0)
            rotated = cv2.warpAffine(self.img, M, (w, h))
            self.rotated_images.append(rotated)
        return self.rotated_images

    def flipping(self, img):
        '''
        flips horizontal, vertical and both
        :param img:
        :return: flipped images list
        '''
        self.flipped_images = []
        originalImage = img
        flipVertical = cv2.flip(originalImage, 0)
        flipHorizontal = cv2.flip(originalImage, 1)
        flipBoth = cv2.flip(originalImage, -1)
        self.flipped_images.append(flipVertical)
        self.flipped_images.append(flipHorizontal)
        self.flipped_images.append(flipBoth)
        return self.flipped_images

    def do_augs(self):
        '''
        calls the flipping and rotating functions
        treat this as a "do all the augs"
        :return: writes images to files as couldn't work out the memory usage nicely.
        '''
        count = 0
        for x in tqdm.tqdm(range(len(self.raw_train))):
            self.img = self.raw_train[x][0]
            api = self.raw_train[x][2]
            rotated = self.rotating(6)
            for rot in rotated:
                flipped = self.flipping(rot)
                for flip in flipped:
                    if self.save_augs is not None:
                        count = count + 1
                        # filename = str(self.save_augs) +\
                        #            '/' + str(self.raw_train[x][1]) +\
                        #            str(count)+ '__' + str(api) + '.jpg'
                        filename = f'{self.save_augs}/{self.raw_train[x][1]}{count}__{api}.jpg'
                        sv_img = fromarray(flip)
                        sv_img.save(filename)
                    else:
                        AttributeError('no save path provided for aug_images')
                        # print(count)
    def save_test_set(self):
        count = 0
        for x in tqdm.tqdm(range(len(self.raw_test))):
            api = self.raw_test[x][2]
            # file_name = self.save_raw + '/' + str(self.raw_test[x][1]) + str(count)\
            #             + '__' + str(api) + '.jpg'
            file_name = f'{self.save_raw}/{self.raw_test[x][1]}{count}__{api}.jpg'
            sv_img = fromarray(self.raw_test[x][0])
            sv_img.save(file_name)
            count = count + 1

    def get_model_images(self):
        self.do_augs()
        self.save_test_set()

    def do_aug_process(self, data):
        img, idx, api = data
        count = 0
        self.img = img
        rotated = self.rotating(6)
        for rot in rotated:
            flipped = self.flipping(rot)
            for flip in flipped:
                if self.save_augs is not None:
                    count = count + 1
                    # filename = str(self.save_augs) +\
                    #            '/' + str(self.raw_train[idx][1]) +\
                    #            str(idx) + '_' + str(count)  + '.jpg'
                    filename = f'{self.save_augs}/{self.raw_train[idx][1]}{idx}_{count}__{api}.jpg'
                    sv_img = fromarray(flip)
                    sv_img.save(filename)
                    # print('done {} / {}'.format(count, len(self.raw_train)))
                else:
                    AttributeError('no save path provided for aug_images')
                    # print(count)


    def do_augs_multi(self):
        data = zip([self.raw_train[x][0] for x in range(len(self.raw_train))],
                   [np.arange(0, len(self.raw_train), 1)[y] for y in range(len(self.raw_train))],
                   [self.raw_train[z][2] for z in range(len(self.raw_train))]
                   )
        for dt in data:
            p = multiprocessing.Process(target=self.do_aug_process, args=(dt,))
            p.start()



