from store.inference.inference import Inference
import os
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt

learner = './code_saves/trained_model_100.pkl'
split_factor = 2

inf = Inference(learner, split_factor)
true, preds = inf.infer()
acc = accuracy_score(true, preds)
print(acc)

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('external_test_set'))
disp.plot()
plt.savefig('./code_saves/ext_test_conf_mtrx.png')
print('done')
