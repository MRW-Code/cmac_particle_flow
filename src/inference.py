import cv2
import torch.cuda
from fastai.vision.all import *
import os
import tqdm
from statistics import mode
from store.models.fastai_prep import FastAIPrep


class Inference:

    def __init__(self, learner, split_factor, test_paths):
        self.split_factor = split_factor
        self.test_pths = self.get_paths(test_paths)
        self.learner = learner


    def get_paths(self, test_paths):
        if test_paths is None:
            test_pths = get_image_files(f'{os.getcwd()}/external_test_set/')
        else:
            test_pths = test_paths
        return test_pths

    def cropping(self, img):
        '''
        splits the big images into smaller quardants
        :param split_factor: is the number to divide the axis by
            num qurdants = split_factor^2
        :return: smaller images (each quardant) of the original image
        '''
        # self.cropped_images = []
        split_factor = self.split_factor + 1
        x = img.shape[0]
        y = img.shape[1]
        points_x = np.linspace(0, x, split_factor, dtype=np.int)
        points_y = np.linspace(0, y, split_factor, dtype=np.int)
        crops = []
        for i in range(len(points_x) - 1):
            for j in range(len(points_y) - 1):
                crops.append(img[points_x[i]:points_x[i] + points_x[1], points_y[j]:points_y[j] + points_y[1]])


        crops = [torch.tensor(crop).cpu() for crop in crops]
        return crops

    def preprocess_split(self):
        for pth in tqdm.tqdm(self.test_pths):
            img_raw = cv2.imread(str(pth))
            stack = self.cropping(img_raw, self.split_factor)
            return stack

    def majority_vote(self, stack):
        vote = []

        for img in stack:
            pred = self.learner.predict(img)
            vote.append(pred)
        ans = mode([i[0] for i in vote])
        return ans

    def infer(self):
        true_labels = []
        pred_labels = []
        api = []
        for pth in tqdm.tqdm(self.test_pths):
            img_raw = torch.tensor(cv2.imread(str(pth))).to('cuda')
            true_labels.append(pth.parent.name)
            # pred_labels.append(self.majority_vote(self.cropping(img_raw)))
            pred_labels.append(self.cropping(img_raw))
            api.append(pth)
        return true_labels, pred_labels, api





