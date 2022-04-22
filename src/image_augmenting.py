import cv2
from tqdm import tqdm
from PIL.Image import fromarray
from joblib import Parallel, delayed
import os
from src.utils import args
from src.helpers import paths_from_dir

class ImageAugmentor():

    def __init__(self, save_path, training_data):
        self.save_path = save_path
        self.raw_train = training_data

    def rotating(self, img, num_rotations):
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        rotated_images = []
        for i in range(num_rotations):
            rotations = list(range(0, 180, num_rotations))
            M = cv2.getRotationMatrix2D(center, rotations[i], 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            rotated_images.append(rotated)
        return rotated_images

    def flipping(self, img):
        flipped_images = []
        originalImage = img
        flipVertical = cv2.flip(originalImage, 0)
        flipHorizontal = cv2.flip(originalImage, 1)
        flipBoth = cv2.flip(originalImage, -1)
        flipped_images.append(flipVertical)
        flipped_images.append(flipHorizontal)
        flipped_images.append(flipBoth)
        return flipped_images

    def do_augs(self):
        if args.from_scratch:
            print('AUGMENTING IMAGES')
            def worker(i):
                count = 0
                path, label = i[1]['fname'], i[1]['label']
                name = path.name
                img = cv2.imread(str(path))
                rotated = self.rotating(img, 6)
                for rot in rotated:
                    flipped = self.flipping(rot)
                    for flip in flipped:
                        count = count + 1
                        filename = f'{self.save_path}/{label}/aug_{count}_{name}'
                        sv_img = fromarray(flip)
                        sv_img.save(filename)
            Parallel(n_jobs=os.cpu_count())(delayed(worker)(i) for i in tqdm(self.raw_train.iterrows(),
                                                                             total=len(self.raw_train)))
        else:
            print('Using images loaded from aug_images dir')

        return None