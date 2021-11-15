import cv2
from fastai.vision.all import *
import os
import tqdm
from statistics import mode

class Inference:

    def __init__(self, learner):
        self.test_pths = self.get_paths()
        self.learner = load_learner(learner)

    def get_paths(self):
        test_pths = get_image_files(os.getcwd() + '/external_test_set/')
        return test_pths

    def cropping(self, img, num_quadrants):
        x = img.shape[0]
        y = img.shape[1]
        if num_quadrants == 4:
            crop1 = torch.tensor(img[0:1745, 0:2680])
            crop2 = torch.tensor(img[1746:x, 0:2680])
            crop3 = torch.tensor(img[0:1745, 2679:y])
            crop4 = torch.tensor(img[1746:x, 2679:y])
        else:
            raise AttributeError('Code only works for 4 quadrants - amend if needed')
        return torch.stack([crop1, crop2, crop3, crop4])

    def preprocess_split(self):
        for pth in tqdm.tqdm(self.test_pths):
            img_raw = cv2.imread(str(pth))
            stack = self.cropping(img_raw, 4)
            return stack

    def majority_vote(self, stack):
        vote = []
        for img in stack:
            vote.append(self.learner.predict(img))
        return mode([i[0] for i in vote])

    def infer(self):
        true_labels = []
        pred_labels = []
        for pth in tqdm.tqdm(self.test_pths):
            img_raw = cv2.imread(str(pth))
            true_labels.append(pth.parent.name)
            pred_labels.append(self.majority_vote(self.cropping(img_raw, 4)))
        return true_labels, pred_labels






