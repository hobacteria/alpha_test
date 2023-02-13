import os
import pandas as pd
import numpy as np
import math
import lightgbm as lgbm
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import datetime
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,  ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Binarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import seaborn as sns
import shap
path = 'C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\'



X_train = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\train_set.csv')
y_train = X_train.pop('TARGET')
X_val = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\val_set.csv')
y_val = X_val.pop('TARGET')
X_test = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\test_set.csv')
y_test = X_test.pop('TARGET')
X_bc = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bc_set.csv')
credit_score = X_bc.pop('credit_score')



def focal_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
    ce = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    weight = y_true * (1 - y_true)
    weight /= np.mean(weight)
    alpha = 0.25
    gamma = 2
    weight = weight * np.power(1 - y_pred, gamma)
    weight = weight * alpha
    weight = weight + (1 - alpha)
    ce = ce * weight
    dL_dp = - (y_true / y_pred) + ((1.0 - y_true) / (1.0 - y_pred))
    dL_dp2 = (y_true / np.power(y_pred, 2)) + ((1.0 - y_true) / np.power(1.0 - y_pred, 2))
    return dL_dp, dL_dp2

def lgb_roc_auc_score(y_true, y_pred):
    return 'roc_auc', roc_auc_score(y_true, y_pred), True

## 
model = lgbm.LGBMClassifier (metric = [lgb_roc_auc_score],
                            objective= focal_cross_entropy,
                            pos_bagging_fraction = 0.8,
                            num_leaves = 128,
                            max_depth = -1,
                            learning_rate = 0.001,
                            n_estimators = 50000,
                            early_stopping_round = 1000,
                            verbose_eval= 0
                            )

ftd_model = model.fit(X_train,y_train,
          eval_set = (X_val,y_val),
          eval_metric = [lgb_roc_auc_score],
          verbose = 1000
          )

#ftd_model.score(train_df,target_df)

sub = ftd_model.predict_proba(X_test)
# train_pred = ftd_model.predict_proba(train_df)

sum(target_df)
sum(sub)


samplesubmission = pd.read_csv(path + 'sample_submission.csv')
samplesubmission['TARGET'] = sub[:,1]
now = datetime.datetime.now()
now = now.strftime('%Y-%m-%d %H%M')

samplesubmission.to_csv(path + f'{now}.csv', index = False)





