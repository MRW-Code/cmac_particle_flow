from inference.inference import Inference
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

learner = '../code_saves/trained_model_200.pkl'
split_factor = 2

inf = Inference(learner, split_factor)
true, preds = inf.infer()
r2 = r2_score(true, preds)
mse = mean_squared_error(true, preds)
mae = mean_absolute_error(true, preds)
print(f'R2 Score = {r2} and MSE = {mse} and MAE = {mae}')
