import pandas as pd
import numpy as np
from inference.inference import Inference
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import os



df = pd.read_csv('./regression/input_and_labels', index_col=0)
print('done')

idx = 3
learner = f'./regression/models/trained_model_{idx}.pkl'
split_factor = 2

inf = Inference(learner, split_factor)
true, preds = inf.infer()
true = list(df.new)
print(f'True = {true}')

acc = accuracy_score(true, preds)
print(f'Accuracy = {acc}')
exit()
cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('../external_test_set'))
disp.plot()
plt.savefig(f'./temp/ext_test_conf_mtrx{idx}.png')

