import pandas as pd
import numpy as np
from fastai.vision.all import *
from src.inference import Inference
import pathlib
from sklearn.metrics import accuracy_score

if __name__ == '__main__':

    fold = 4
    # mean_acc = []

    model_path = f'./checkpoints/ttv_kfold_no_leak/' +\
                   f'models/swinv2_base_window12to24_192to384_22kft1k/' +\
                   f'sf_7_bs8_accum4/fold_{fold}.pkl'
    items_path = pd.read_csv(f'./pred_csv/ttv_kfold_no_leak/7/external_test_majority_fold_{fold}.csv',
                             index_col=0)
    X_test = items_path.api
    X_test = [pathlib.Path(x) for x in items_path.api]
    model = load_learner(model_path, cpu=False)
    do_inference = Inference(model, 7, X_test)
    true_labels, pred_labels, api = do_inference.infer_majority()
    maj_acc = accuracy_score(true_labels, [label[0] for label in pred_labels])
    print(maj_acc)
    # mean_acc.append(maj_acc)
    maj_df = pd.DataFrame({'true': true_labels,
                           'preds': pred_labels,
                           'api': api})
    maj_df.to_csv(f'./scores_{fold}.csv')
    print('done')
    # print(np.mean(maj_acc))



