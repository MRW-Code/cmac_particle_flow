import pandas as pd
import numpy as np
from inference.inference import Inference
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os


co_ez = 4
ez_fr = 10

def relabel_val(co_ez, ez_fr, true, api):
    ref_df = pd.read_csv('./regression/dump/FFc_data.csv')
    new_labels = [None] * len(true)
    for idx, label in enumerate(true):
        api_test = api[idx].stem
        ffc = ref_df['FFc'][ref_df['api'] == api_test].values[0]

        # print(api[idx], ffc)

        if ffc <= co_ez:
            new_labels[idx] = 'cohesive'
        elif co_ez < ffc < ez_fr:
            new_labels[idx] = 'EasyFlowing'
        else:
            new_labels[idx] = 'FreeFlowing'
    return new_labels



idx = co_ez	
learner = f'./regression/models/trained_model_{idx}.pkl'
print(learner)
split_factor = 2

inf = Inference(learner, split_factor)
true, preds, api = inf.infer()

new_true = relabel_val(co_ez, ez_fr, true, api)
print(new_true)
print(preds)

acc = accuracy_score(new_true, preds)
print(f'Accuracy = {acc}')

cm = confusion_matrix(new_true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('./external_test_set'))
disp.plot()
plt.savefig(f'./ext_test_conf_mtrx{idx}.png')

