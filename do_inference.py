from inference.inference import Inference
import os
from sklearn.metrics import accuracy_score


learner = '/home/matthew/PycharmProjects/cmac_particle_flow/' \
          'code_saves/trained_model_200.pkl'

inf = Inference(learner)
true, preds = inf.infer()
acc = accuracy_score(true, preds)
print(acc)


print('done')