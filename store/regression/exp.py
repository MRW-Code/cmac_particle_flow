import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
sys.path.append('../..')

import torch
import glob
torch.cuda.empty_cache()
from store.inference import Inference
from fastai.vision.all import *
from fastai.distributed import *
import os
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

def empty_file(path):
    files = glob.glob(path + '/*')
    for file in files:
        os.remove(file)


def relabel(co_ez, ez_fr, df):
    new_labels = [None] * len(df.FFc)
    for idx, label in enumerate(df.FFc):
        if label <= co_ez:
            new_labels[idx] = 'cohesive'
        elif co_ez < label < ez_fr:
            new_labels[idx] = 'EasyFlowing'
        else:
            new_labels[idx] = 'FreeFlowing'
    df['new'] = new_labels
    return df

def relabel_val(co_ez, ez_fr, true, api):
    ref_df = pd.read_csv('dump/FFc_data.csv')
    new_labels = [None] * len(true)
    for idx, label in enumerate(true):
        api_test = api[idx]
        ffc = ref_df[ref_df['api'] == api[idx], 'FFc']
        print(api[idx], ffc)

        if ffc <= co_ez:
            new_labels[idx] = 'cohesive'
        elif co_ez < ffc < ez_fr:
            new_labels[idx] = 'EasyFlowing'
        else:
            new_labels[idx] = 'FreeFlowing'
    return new_labels

def run(idx_list):
    my_indexes = []
    val_acc = []
    for idx in idx_list:
        torch.cuda.empty_cache()

        # empty_file('../aug_images')
        # empty_file('../split_test_images')

        # # change the second number to the appropriate split index
        # prep = RegressionFastAIPrep('../images', 0, 2, '../aug_images', '../split_test_images',
        #                             multi=True,
        #                             oversample=False)
        # prep.check_test_train_data()
        # df = prep.get_fastai_df()
        # # df.to_csv('info.csv')
        # print('done')

        co_ez = idx
        ez_fr = 10

        # df = relabel(co_ez, ez_fr, df)
        # df.to_csv('input_and_labels')

        # tfms = None
        # dls = ImageDataLoaders.from_df(df,
        #                                seed=None,
        #                                fn_col=0,
        #                                folder=None,
        #                                suff='',
        #                                label_col=3,
        #                                label_delim=None,
        #                                valid_col=1,
        #                                item_tfms=None,
        #                                batch_tfms=None,
        #                                bs=16,
        #                                val_bs=None,
        #                                shuffle=True,
        #                                device=None)

        # learn = cnn_learner(dls, resnet18, metrics=[accuracy]).to_fp16()

        # learn.fine_tune(1, 0.00001, cbs=[SaveModelCallback(fname=f'./best_cbs_relab_{idx}'),
        #                                    ReduceLROnPlateau(monitor='valid_loss',
        #                                                      min_delta=0.1,
        #                                                      patience=2)])

        # # for training across multiple gpus
        # # with learn.distrib_ctx(sync_bn=False): learn.fine_tune(100, 0.0001, cbs=[SaveModelCallback(fname='./best_cbs'),
        # #                                  ReduceLROnPlateau(monitor='valid_loss',
        # #                                                    min_delta=0.1, patience=2)])

        # learn.export(f'./models/trained_model_{idx}.pkl')

        # learn.recorder.plot_loss()
        # plt.savefig(f'./training_plot_{idx}.png')

        # interp = ClassificationInterpretation.from_learner(learn)
        # interp.plot_confusion_matrix()
        # plt.savefig(f'./dump/conf_mtrx_train{idx}.png')
    
        learner = f'./models/trained_model_{idx}.pkl'
        split_factor = 2
    
        inf = Inference(learner, split_factor)
        true, preds, api = inf.infer()
        print(true)

        true = relabel_val(co_ez, ez_fr, true, api)

        print(f'True = {true}')
        exit()
        acc = accuracy_score(true, preds)
        print(f'Accuracy = {acc}')

        cm = confusion_matrix(true, preds)
        disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('../external_test_set'))
        disp.plot()
        plt.savefig(f'./dump/ext_test_conf_mtrx{idx}.png')
    
        my_indexes.append(idx)
        val_acc.append(acc)
    print(my_indexes, val_acc)


# print(sys.path)
idx_list = [3,4,5,6,7]
test = run(idx_list)
print('done')
