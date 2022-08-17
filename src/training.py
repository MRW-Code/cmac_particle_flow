import os

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from fastai.vision.all import *
from src.utils import args
from src.image_augmenting import ImageAugmentor
from src.helpers import paths_from_dir
from src.image_splitting import ImageSplitter
from src.inference import Inference
import timm


def train_fastai_model_classification(model_df, count, exp_type):
    dls = ImageDataLoaders.from_df(model_df,
                                   fn_col=0,
                                   label_col=1,
                                   valid_col=2,
                                   item_tfms=None,
                                   batch_tfms=None,
                                   y_block=CategoryBlock(),
                                   bs=args.batch_size,
                                   shuffle=True)

    metrics = [error_rate, accuracy]
    learn = vision_learner(dls, args.model, metrics=metrics).to_fp16()
    if args.grad_accum == 1:
        learn.fine_tune(100, cbs=[SaveModelCallback(monitor='valid_loss', fname=f'./csd_{args.no_augs}_best_cbs.pth'),
                                ReduceLROnPlateau(monitor='valid_loss',
                                                  min_delta=0.05,
                                                  patience=2),
                                 EarlyStoppingCallback(monitor='accuracy', min_delta=0.1, patience=20)])
    else:
        learn.fine_tune(100, cbs=[SaveModelCallback(monitor='valid_loss', fname=f'./csd_{args.no_augs}_best_cbs.pth'),
                                ReduceLROnPlateau(monitor='valid_loss',
                                                  min_delta=0.05,
                                                  patience=2),
                                 EarlyStoppingCallback(monitor='accuracy', min_delta=0.1, patience=5),
                                  GradientAccumulation(n_acc=args.grad_accum)])

    # print(learn.validate())
    ### CHANGE THIS SAVE PATH
    os.makedirs(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}', exist_ok=True)
    learn.export(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}/fold_{count}.pkl')

    if args.make_figs:
        os.makedirs(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}', exist_ok=True)
        learn.recorder.plot_loss()
        plt.savefig(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}_loss_plot_{count}.png',
                    bbox='tight')

        interp = ClassificationInterpretation.from_learner(learn)
        interp.plot_confusion_matrix()
        plt.savefig(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}_confmat_{count}.png')

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
        model = load_learner(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}/fold_{count}.pkl', cpu=False)
        best_metrics.append(model.final_record)
        count += 1

    print(best_metrics)
    print(f'mean loss = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
    print(f'mean acc = {np.mean([best_metrics[x][3] for x in range(n_splits)])}')
    return None

def split_first_model(n_splits, img_paths):
    paths = img_paths
    labels = [pth.parent.name for pth in paths]

    count = 0
    best_metrics = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)
    for train_index, val_index in tqdm(kfold.split(paths, labels)):

        X_train, X_val = np.array(paths)[train_index], np.array(paths)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]

        # Do the image splitting
        splitter = ImageSplitter(img_paths=None, split_factor=args.split_factor, val_idx=None)
        splitter.save_split_first(X_train, X_val)

        # Get train/val of the new split images
        X_train = paths_from_dir('./split_images/train')
        y_train = [pth.parent.name for pth in X_train]
        X_val = paths_from_dir('./split_images/valid')
        y_val = [pth.parent.name for pth in X_val]

        # Make a model df
        train_df = pd.DataFrame({'fname': X_train, 'label': y_train})
        train_df.loc[:, 'is_valid'] = 0
        val_df = pd.DataFrame({'fname': X_val, 'label': y_val})
        val_df.loc[:, 'is_valid'] = 1

        # Apply Augs if needed and remake model df if applied
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

        exp_type = 'split_first'
        trainer = train_fastai_model_classification(model_df, count, exp_type=exp_type)
        model = load_learner(f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}/fold_{count}.pkl', cpu=False)
        best_metrics.append(model.final_record)
        count += 1

    print(best_metrics)
    print(f'mean loss = {np.mean([best_metrics[x][2] for x in range(n_splits)])}')
    print(f'mean acc = {np.mean([best_metrics[x][3] for x in range(n_splits)])}')
    return None


def ttv_model(img_paths):
    paths = img_paths
    labels = [pth.parent.name for pth in paths]

    X_train, X_val, y_train, y_val = train_test_split(paths, labels,
                                                        test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                        test_size=0.5, random_state=0)

    # Do the image splitting
    splitter = ImageSplitter(img_paths=None, split_factor=args.split_factor, val_idx=None)
    splitter.save_split_first(X_train, X_val)

    # Get train/val of the new split images
    X_train = paths_from_dir('./split_images/train')
    y_train = [pth.parent.name for pth in X_train]
    X_val = paths_from_dir('./split_images/valid')
    y_val = [pth.parent.name for pth in X_val]

    # Make a model df
    train_df = pd.DataFrame({'fname': X_train, 'label': y_train})
    train_df.loc[:, 'is_valid'] = 0
    val_df = pd.DataFrame({'fname': X_val, 'label': y_val})
    val_df.loc[:, 'is_valid'] = 1

    # Apply Augs if needed and remake model df if applied
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

    exp_type = 'ttv'
    trainer = train_fastai_model_classification(model_df, 0, exp_type=exp_type)
    model = load_learner(
        f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}/fold_{0}.pkl',
        cpu=False)

    # majority vote
    do_inference = Inference(model, args.split_factor, X_test)
    true_labels, pred_labels, api = do_inference.infer_majority()
    acc = accuracy_score(true_labels, pred_labels)
    print(f'majority vote acc = {acc}')

    # Not majority
    true_labels, pred_labels, api = do_inference.infer_single()
    acc = accuracy_score(true_labels, pred_labels)
    print(f'single vote acc = {acc}')

    return None

def kfold_ttv_model(n_splits, img_paths, test_pct):
    paths = img_paths
    labels = [pth.parent.name for pth in paths]

    count = 0
    best_val_metrics = []
    best_test_single = []
    best_test_majority = []
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True)

    for train_index, val_index in tqdm(kfold.split(paths, labels)):

        X_train, X_val = np.array(paths)[train_index], np.array(paths)[val_index]
        y_train, y_val = np.array(labels)[train_index], np.array(labels)[val_index]
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val,
                                                        test_size=0.5, random_state=0)



        # Do the image splitting
        splitter = ImageSplitter(img_paths=None, split_factor=args.split_factor, val_idx=None)
        splitter.save_split_first(X_train, X_val)

        # Get train/val of the new split images
        X_train = paths_from_dir('./split_images/train')
        y_train = [pth.parent.name for pth in X_train]
        X_val = paths_from_dir('./split_images/valid')
        y_val = [pth.parent.name for pth in X_val]

        # Make a model df
        train_df = pd.DataFrame({'fname': X_train, 'label': y_train})
        train_df.loc[:, 'is_valid'] = 0
        val_df = pd.DataFrame({'fname': X_val, 'label': y_val})
        val_df.loc[:, 'is_valid'] = 1

        # Apply Augs if needed and remake model df if applied
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

        exp_type = 'ttv_kfold'
        trainer = train_fastai_model_classification(model_df, count, exp_type=exp_type)
        model = load_learner(
            f'./checkpoints/{exp_type}/models/{args.model}/sf_{args.split_factor}_bs{args.batch_size}_accum{args.grad_accum}/fold_{count}.pkl',
            cpu=False)
        best_val_metrics.append(model.final_record)

        # majority vote
        do_inference = Inference(model, args.split_factor, X_test)
        true_labels, pred_labels, api = do_inference.infer_majority()
        maj_acc = accuracy_score(true_labels, pred_labels)
        best_test_majority.append(maj_acc)

        # Not majority
        true_labels, pred_labels, api = do_inference.infer_single()
        single_acc = accuracy_score(true_labels, pred_labels)
        best_test_single.append(single_acc)

        print(f'fold {count}, mean loss = {np.mean([best_val_metrics[x][2] for x in range(count+1)])}')
        print(f'fold {count}, mean val acc = {np.mean([best_val_metrics[x][3] for x in range(count+1)])}')
        print(f'fold {count}, mean maj test acc = {np.mean(best_test_majority)}')
        print(f'fold {count}, mean single test acc = {np.mean(best_test_single)}')


        count += 1

    # print(best_metrics)
    print(f'mean loss = {np.mean([best_val_metrics[x][2] for x in range(n_splits)])}')
    print(f'mean val acc = {np.mean([best_val_metrics[x][3] for x in range(n_splits)])}')
    print(f'mean maj test acc = {np.mean(best_test_majority)}')
    print(f'mean single test acc = {np.mean(best_test_single)}')
    return None

