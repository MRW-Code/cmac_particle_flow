from inference.inference import Inference
import os
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
from relabel_main import co_ez, ez_fr
learner = './dump/trained_model_100.pkl'
split_factor = 2

inf = Inference(learner, split_factor)
true, preds = inf.infer()

def relabel(co_ez, ez_fr, true):
    new_labels = [None] * len(true)
    for idx, label in enumerate(true):
        if label <= co_ez:
            new_labels[idx] = 'Cohesive'
        elif co_ez < label < ez_fr:
            new_labels[idx] = 'EasyFlowing'
        else:
            new_labels[idx] = 'FreeFlowing'
    return true

true = get_true(co_ez, ez_fr, true)


acc = accuracy_score(true, preds)
print(acc)

cm = confusion_matrix(true, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=os.listdir('./external_test_set'))
disp.plot()
plt.savefig('./code_saves/ext_test_conf_mtrx.png')
print('done')
