import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix
import numpy as np
from src.utils import args

df_list = os.listdir(f'pred_csv/ttv_kfold_best')
print(df_list)

## for majority vote
total = np.zeros((3,3))
print(total)
for file in df_list:
    if 'single' in file:
        pass
    else:
        df = pd.read_csv(f'pred_csv/ttv_kfold_best/{args.split_factor}/{file}', index_col=0)
        labels = sorted(list(df.true.value_counts().index))
        conf_mat = confusion_matrix(df.true, df.preds, labels=labels)
        print(conf_mat)
        print(df.true.value_counts())
        total += conf_mat
        print(total)
        print('')
        print('')


figure = sns.heatmap(total, annot=True, cmap = 'Blues', xticklabels=labels, yticklabels=labels)
plt.show()


## for single vote
total = np.zeros((3,3))
print(total)
for file in df_list:
    if 'majority' in file:
        pass
    else:
        df = pd.read_csv(f'pred_csv/ttv_kfold_best/{args.split_factor}/{file}', index_col=0)
        labels = sorted(list(df.true.value_counts().index))
        conf_mat = confusion_matrix(df.true, df.preds, labels=labels)
        print(conf_mat)
        print(df.true.value_counts())
        total += conf_mat
        print(total)
        print('')
        print('')


figure = sns.heatmap(total, annot=True, cmap = 'Blues', xticklabels=labels, yticklabels=labels,
                     fmt='g')
plt.show()