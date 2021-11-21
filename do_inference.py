from inference.inference import Inference
import os
from sklearn.metrics import accuracy_score


learner = './code_saves/trained_model_100.pkl'
split_factor = 2

inf = Inference(learner, split_factor)
true, preds = inf.infer()
acc = accuracy_score(true, preds)
print(acc)


print('done')
