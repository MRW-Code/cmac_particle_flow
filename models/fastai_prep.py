from augmentations.image_augmenting import ImageAugmentor
import pandas as pd
import os
import re

class FastAIPrep(ImageAugmentor):

    def __init__(self, data_path, split_idx, split_factor, save_train, save_test, multi, oversample):
        super().__init__(data_path, split_idx, split_factor, save_train, save_test, oversample)
        self.train_save_path = save_train
        self.test_save_path = save_test
        self.multi = multi

    def check_train_dir(self):
        print(self.train_save_path)
        print(len(os.listdir(self.train_save_path)))
        if len(os.listdir(self.train_save_path)) == 0:
            if self.multi == True:
                self.do_augs_multi()
                print('loaded train images')
            else:
                self.do_augs()
                print('loaded train images')
        else:
            print('Using pre-saved train images')


    def check_test_dir(self):
        print(self.test_save_path)
        print(len(os.listdir(self.test_save_path)))
        if len(os.listdir(self.test_save_path)) == 0:
            self.save_test_set()
            print('loaded test images')
        else:
            print('Using pre-saved test images')

    def check_test_train_data(self):
        self.check_test_dir()
        self.check_train_dir()
        print('Check complete, there should be data now')

    def get_df_attributes(self, direc, is_valid):
        df = pd.DataFrame({'fname' : os.listdir(direc)})
        df['fname'] = direc + '/' + df['fname']
        df['labels'] = [re.search(r's\/(\D*)', x).group(1) for x in df['fname']]
        df['is_valid'] = is_valid
        return df

    def get_fastai_df(self):
        train = self.get_df_attributes(self.train_save_path, is_valid=0)
        val = self.get_df_attributes(self.test_save_path, is_valid=1)
        df = pd.concat([train, val], axis=0)
        return df