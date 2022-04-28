import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from fastai.vision.all import *
from src.utils import args
from src.image_augmenting import ImageAugmentor
from src.helpers import paths_from_dir

def train_fastai_model_classification(model_df, count, exp_type):
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=CategoryBlock(),
                                   bs=32,
                                   shuffle=True)

    metrics = [error_rate, accuracy]
    learn = cnn_learner(dls, resnet18, metrics=metrics).to_fp16()
    learn.fine_tune(50, cbs=[SaveModelCallback(monitor='accuracy', fname=f'./csd_{args.no_augs}_best_cbs.pth'),
                            ReduceLROnPlateau(monitor='valid_loss',
                                              min_delta=0.1,
                                              patience=2),
                             EarlyStoppingCallback(monitor='accuracy', min_delta=0.1, patience=10)])

    # print(learn.validate())
    ### CHANGE THIS SAVE PATH
    learn.export(f'./checkpoints/{exp_type}/models/sf{args.split_factor}_fold_{count}.pkl')


def kfold_model(n_splits):
    paths = paths_from_dir('./split_images/train')
    labels = [pth.parent.name for pth in paths]

    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    count = 0
    best_metrics = []
    for train_index, val_index in tqdm(kfold.split(paths, labels)):
        X_train, X_val = np.array(paths)[train_index], np.array(paths)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

        train_df = pd.DataFrame({'fname': X_train, 'label': y_train})
        train_df.loc[:, 'is_valid'] = 0
        val_df = pd.DataFrame({'fname': X_val, 'label': y_val})
        val_df.loc[:, 'is_valid'] = 1

        if args.no_augs:
            model_df = pd.concat([train_df, val_df])
        else:
            raw_model_df = pd.concat([train_df, val_df])
            augmentor = ImageAugmentor(save_path='./aug_images', training_data=raw_model_df)
            augmentor.do_augs()
            aug_paths = paths_from_dir('./aug_images')
            aug_labels = [j.parent.name for j in aug_paths]
            aug_df = pd.DataFrame({'fname': aug_paths, 'label': aug_labels})
            aug_df.loc[:, 'is_valid'] = 0
            model_df = pd.concat([aug_df, val_df])

        exp_type = 'splitting_test'
        trainer = train_fastai_model_classification(model_df, count, exp_type=exp_type)
        model = load_learner(f'./checkpoints/{exp_type}/models/sf{args.split_factor}_fold_{count}.pkl', cpu=False)
        best_metrics.append(model.final_record)
        count += 1

    print(best_metrics)
    print(f'mean acc = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
    return None

