import numpy as np
import pandas as pd
import optuna
from lightgbm import LGBMClassifier
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import Binarizer
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,  ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier


path = 'C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\'



X_train = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\train_set.csv')
y_train = X_train.pop('TARGET')
X_val = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\val_set.csv')
y_val = X_val.pop('TARGET')
X_test = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\test_set.csv')
y_test = X_test.pop('TARGET')
X_bc = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bc_set.csv')
credit_score = X_bc.pop('credit_score')

y_train = y_train.squeeze()

X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]
X_bc = X_bc[X_train.columns]

X = pd.concat([X_train,X_val])
y = pd.concat([y_train,y_val])

def lgb_f1_score(y_true, y_pred):
    y_pred_proba =  y_pred
    y_pred = Binarizer(threshold = 0.2).fit_transform(y_pred_proba.reshape(-1,1))

    return 'f1', f1_score(y_true, y_pred), True

## lgbm 튜닝
def lgbm_objective(trial: Trial) -> float:
    
    params_lgb = {
        "random_state": 4885,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "binary",
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 3e-5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 9e-2),
        "max_depth": trial.suggest_int("max_depth", 1, 20),
        "num_leaves": trial.suggest_int("num_leaves", 2, 256),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
        "subsample": trial.suggest_float("subsample", 0.3, 1.0),
        "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "max_bin": trial.suggest_int("max_bin", 200, 500)
    }
    
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X, y, test_size=0.2)

    model = LGBMClassifier(**params_lgb)
    model.fit(
        X_train_,
        y_train_,
        eval_metric = "auc",
        eval_set=[(X_train_, y_train_),(X_valid_, y_valid_)],
        early_stopping_rounds=100,
        verbose=False
    )

    lgb_pred = model.predict_proba(X_valid_)[:,1]
    score = roc_auc_score(y_valid_, lgb_pred)
    
    return score


sampler = TPESampler(seed=4885)
study = optuna.create_study(
    study_name="lgbm_parameter_opt",
    direction="maximize",
    sampler=sampler
)

study.optimize(lgbm_objective, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)


params_lgb = {'reg_alpha': 2.787007800132633e-05,
              'reg_lambda': 0.0032531965015774405,
              'max_depth': 12, 'num_leaves': 207,
              'colsample_bytree': 0.5004712058394702,
              'subsample': 0.6869320257998226,
              'subsample_freq': 8,
              'min_child_samples': 36,
              'max_bin': 256}

model = LGBMClassifier (**params_lgb)

ftd_model = model.fit(
          X = X_train, y = y_train,
          eval_metric = "auc",
          eval_set = (X_val,y_val),
          early_stopping_rounds=100,
          verbose=False
          )

test_score = ftd_model.predict_proba(X_test)[:,1]
test_score_bi = Binarizer(threshold = 0.1).fit_transform(test_score.reshape(-1,1))
roc_auc_score(y_test,test_score)



## mlp 튜닝
def mlp_objective(trial: Trial) -> float:
    
    params_lgb = {
        'activation' : 'relu',
        'hidden_layer_sizes' : trial.suggest_int('n_layers', 1,1000)
    }
    
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X, y, test_size=0.2)
    model = MLPClassifier(**params_lgb)
    model.fit(
        X_train_,
        y_train_
    )
    pred = model.predict_proba(X_valid_)[:,1]
    score = roc_auc_score(y_valid_, pred)
    
    return score


sampler = TPESampler(seed=4885)
study = optuna.create_study(
    study_name="mlp_parameter_opt",
    direction="maximize",
    sampler=sampler
)

study.optimize(mlp_objective, n_trials=10)
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)

params_lgb = {
        'activation' : 'relu',
        'n_layers': 57}

## rf 튜닝
def RF_objective(trial: Trial) -> float:
    max_depth = trial.suggest_int('max_depth', 1, 256)
    max_leaf_nodes = trial.suggest_int('max_leaf_nodes', 2, 1000)
    n_estimators =  trial.suggest_int('n_estimators', 100, 1500)
    
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X, y, test_size=0.2)
   
    model = RandomForestClassifier(max_depth = max_depth, max_leaf_nodes = max_leaf_nodes,n_estimators = n_estimators,n_jobs=-1,random_state=4885)

    model.fit(X_train_, y_train_)    
    pred = model.predict_proba(X_valid_)[:,1]
    score = roc_auc_score(y_valid_, pred)

    return score

sampler = TPESampler(seed=4885)
rf_study = optuna.create_study(
    study_name="rf_parameter_opt",
    direction="maximize",
    sampler=sampler
)


rf_study.optimize(RF_objective, n_trials=10)
print("Best Score:", rf_study.best_value)
print("Best trial:", rf_study.best_trial.params)


rf_params = {'max_depth' : 255, 
                'max_leaf_nodes' : 762,
                'n_estimators' : 316,
                'n_jobs' :-1 ,
                'random_state' : 4885}


## ada

def ada_objective(trial):
    
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.001, 1.0, log=True)
    
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=1)
   
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X, y, test_size=0.2)
   
    model.fit(X_train_, y_train_)
    pred = model.predict_proba(X_valid_)[:,1]
    score = roc_auc_score(y_valid_, pred)
    return score


sampler = TPESampler(seed=4885)
ada_study = optuna.create_study(
    study_name="ada_parameter_opt",
    direction="maximize",
    sampler=sampler
)


ada_study.optimize(ada_objective, n_trials=10)
print("Best Score:", ada_study.best_value)
print("Best trial:", ada_study.best_trial.params)
