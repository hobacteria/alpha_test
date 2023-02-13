import os
import pandas as pd
import numpy as np
import math
import lightgbm as lgb
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

y_train = y_train.squeeze()

X_val = X_val[X_train.columns]
X_test = X_test[X_train.columns]
X_bc = X_bc[X_train.columns]


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
    return np.mean(ce)

def focal_cross_entropy_lgb(y_true, y_pred):
    return 'focal_cross_entropy', focal_cross_entropy(y_true, y_pred)

def lgb_roc_auc_score(y_true, y_pred):
    return 'roc_auc', roc_auc_score(y_true, y_pred), True

### 스태킹 작업을 위한 함수
## 변수명 재설정

def get_stacking(model,X_train,y_train,X_val,X_test,bc_train,n_folds = 10):
    
    kfold = StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=4885)
    train_fold_predict = np.zeros((X_train.shape[0],1))
    val_predict = np.zeros((X_val.shape[0],n_folds))
    test_predict = np.zeros((X_test.shape[0],n_folds))
    bc_predict = np.zeros((bc_train.shape[0],n_folds))
    
    print("model : ", model.clf.__class__.__name__)
    
    for cnt, (train_index, valid_index) in tqdm(enumerate(kfold.split(X_train,y_train)),total= n_folds):
        X_train_ = X_train.iloc[train_index]
        y_train_ = y_train.iloc[train_index]
        X_validation = X_train.iloc[valid_index]
        #print(f'\n {model.clf.__class__.__name__} training {cnt + 1}th fold')
        
        model.fit(X_train_, y_train_.squeeze())

        #해당 폴드에서 학습된 모델에다가 검증 데이터(X_validation)로 예측 후 저장
        ## proba 예측이라 *,2 shape이기 때문에 [:,1] 인덱싱을 해줌
        train_fold_predict[valid_index, :] = model.predict(X_validation)[:,1].reshape(-1, 1)

        #해당 폴드에서 생성된 모델에게 원본 테스트 데이터(X_test)를 이용해서 예측을 수행하고 저장

        val_predict[:, cnt] = model.predict(X_val)[:,1]

        test_predict[:, cnt] = model.predict(X_test)[:,1]

        bc_predict[:, cnt] = model.predict(bc_train)[:,1]

    #for문이 끝나면 test_pred는 평균을 내서 하나로 합친다.

    val_predict_mean = np.mean(val_predict, axis =1).reshape(-1, 1)
    test_predict_mean = np.mean(test_predict, axis =1).reshape(-1, 1)
    bc_predict_mean = np.mean(bc_predict, axis =1).reshape(-1, 1)

    return train_fold_predict, val_predict_mean, test_predict_mean, bc_predict_mean


## 이번엔 1번째 앙상블 모델을 생성하는 과정
## 학습을 도울 핼퍼클래스 생성 (train, fit이 혼용되어 있기 때문에 통일 시켜주는 과정)
class SKlearnHelper(object):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)
    
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
    
    def predict(self, x):
        return self.clf.predict_proba(x)
    
    def fit(self, x, y):
        return self.clf.fit(x, y)
    
    def feature_importances(self, x, y):
        return self.clf.fit(x, y).feature_importances_

## 파라미터들. 임의로 정한 파라미터이고 나중에 성능이 생각보다 안나오면 optuna를 사용할 것... 할게 많다
rf_params = {'max_depth' : 255, 
                'max_leaf_nodes' : 762,
                'n_estimators' : 316,
                'n_jobs' :-1 ,
                'random_state' : 4885}

# Extra Trees
et_params = {
    'criterion' : 'entropy',
    'n_jobs': -1,
    'n_estimators': 1000,
    'max_depth': 8,
    'min_samples_leaf': 2,
    'verbose': 0
}

# AdaBoost
ada_params = {
'n_estimators': 169, 
'learning_rate': 0.8731361854861802
}


# Support Vector
svc_params = {
    'kernel': 'linear',
    'C': 0.025,
    'probability' : True
}

##LogisticRegression
## 설정할게 딱히..?
lr_params = {
    'max_iter' : 5000
}

## MLP
MLP_params = {
        'activation' : 'relu',
        'hidden_layer_sizes': 57}

lgbm_params = {'objective': focal_cross_entropy_lgb,
               'metric':[lgb_roc_auc_score],
              'reg_alpha': 2.787007800132633e-05,
              'reg_lambda': 0.0032531965015774405,
              'max_depth': 12, 'num_leaves': 207,
              'colsample_bytree': 0.5004712058394702,
              'subsample': 0.6869320257998226,
              'subsample_freq': 8,
              'min_child_samples': 36,
              'max_bin': 256}

## 시드
SEED = 4885

#svc = SKlearnHelper(clf=SVC, seed=SEED, params=svc_params)
rf = SKlearnHelper(clf=RandomForestClassifier, seed=SEED, params=rf_params)
et = SKlearnHelper(clf=ExtraTreesClassifier, seed=SEED, params=et_params)
lgbm = SKlearnHelper(clf=lgb.LGBMClassifier, seed=SEED, params=lgbm_params)
ada = SKlearnHelper(clf=AdaBoostClassifier, seed=SEED, params=ada_params)
#gb = SKlearnHelper(clf=GradientBoostingClassifier, seed=SEED, params=gb_params)
lr = SKlearnHelper(clf=LogisticRegression, seed=SEED, params=lr_params)
MLP = SKlearnHelper(clf=MLPClassifier, seed=SEED, params = MLP_params)


'''
smotenc는 bassline에서 실험 해 본 결과 오히려 더 낮은 성능을 나타냄
cv 스태킹 특성상 높은 시간까지 소요되는 바, 사용하지 않기로 결정
'''



## 각 모델별 학습 시작
## svm 학습속도가 너무 오래걸림. 패스

#svm_train, svm_test = get_stacking(svc, X_train, y_train,X_val, X_test)
lgbm_res= get_stacking(lgbm, X_train, y_train,X_val, X_test,X_bc)
ada_res= get_stacking(ada, X_train, y_train,X_val, X_test,X_bc)
#gb_train, gb_test = get_stacking(gb, X_train, y_train, X_val,X_test)
mlp_res = get_stacking(MLP, X_train, y_train, X_val,X_test,X_bc)
rf_res= get_stacking(rf, X_train, y_train,X_val, X_test,X_bc)
et_res = get_stacking(et, X_train, y_train,X_val, X_test,X_bc)
lr_res = get_stacking(lr, X_train, y_train, X_val,X_test,X_bc)

## 최종 모델
X_train_new = pd.DataFrame(np.zeros((X_train.shape[0],6)))
X_val_new = pd.DataFrame(np.zeros((X_val.shape[0],6)))
X_test_new = pd.DataFrame(np.zeros((X_test.shape[0],6)))
X_bc_new = pd.DataFrame(np.zeros((X_bc.shape[0],6)))

# X_train_new['svm_train'] ,X_test_new['svm_test '] = svm_train, svm_test 
#X_train_new.iloc[:,3] ,X_val_new.iloc[:,0],X_test_new.iloc[:,3] = gb_train, gb_test 
X_train_new.iloc[:,0] ,X_val_new.iloc[:,0],X_test_new.iloc[:,0], X_bc_new.iloc[:,0] = mlp_res
X_train_new.iloc[:,1] ,X_val_new.iloc[:,1],X_test_new.iloc[:,1], X_bc_new.iloc[:,1] = rf_res
X_train_new.iloc[:,2] ,X_val_new.iloc[:,2],X_test_new.iloc[:,2], X_bc_new.iloc[:,2] = et_res
X_train_new.iloc[:,3] ,X_val_new.iloc[:,3],X_test_new.iloc[:,3], X_bc_new.iloc[:,3] = lr_res
X_train_new.iloc[:,4] ,X_val_new.iloc[:,4],X_test_new.iloc[:,4], X_bc_new.iloc[:,4] = lgbm_res
X_train_new.iloc[:,5] ,X_val_new.iloc[:,5],X_test_new.iloc[:,5], X_bc_new.iloc[:,5] = ada_res


## 검증 데이터는 타겟 데이터의 빈도가 동일한 데이터를 사용. 언더 샘플링.

## 메타모델 lgbm


def lgb_roc_auc_score(y_true, y_pred):
    return 'roc_auc', roc_auc_score(y_true, y_pred), True


## 메타모델 최적화

def lgbm_objective(trial: Trial) -> float:
    
    params_lgb = {
        "random_state": 4885,
        "verbosity": -1,
        'metric' : 'cross_entropy_lambda',
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
    
    X_train_, X_valid_, y_train_, y_valid_ = train_test_split(X_train_new, y_train, test_size=0.2)

    model = lgb.LGBMClassifier(**params_lgb)
    model.fit(
        X_train_,
        y_train_,
        eval_metric = lgb_roc_auc_score,
        eval_set=[(X_train_, y_train_),(X_valid_, y_valid_)],
        early_stopping_rounds=100,
        verbose=False
    )

    lgb_pred = model.predict_proba(X_valid_)[:,1]
    score = roc_auc_score(y_valid_, lgb_pred)
    
    return score


sampler = TPESampler(seed=4885)
meta_study = optuna.create_study(
    study_name="lgbm_parameter_opt",
    direction="maximize",
    sampler=sampler
)

meta_study.optimize(lgbm_objective, n_trials=100)
print("Best Score:", meta_study.best_value)
print("Best trial:", meta_study.best_trial.params)

meta_params = dict({
        "random_state": 4885,
        "verbosity": -1,
        "learning_rate": 0.05,
        "n_estimators": 10000,
        "objective": "binary",
        'metric' : 'binary'
        }, **meta_study.best_trial.params)


## 파라미터로 학습 시작
model = lgb.LGBMClassifier (**meta_params)

ftd_model = model.fit(
            X_train_new,
            y_train,
            eval_metric = 'binary',
            eval_set=[(X_train_new, y_train),(X_val_new, y_val)],
            early_stopping_rounds=100,
            verbose=True
          )


from sklearn.metrics import f1_score


lgbm_sub = ftd_model.predict_proba(X_test_new)
lgbm_sub = ftd_model.predict_proba(X_test_new)
ftd_model.score(X_test_new,y_test)

## 메타모델 MLP
mlp_meta = MLPClassifier(50,random_state = SEED,activation = 'relu')
mlp_meta_ftd = mlp_meta.fit(X = X_train_new,y = y_train)
mlp_meta_sub = mlp_meta_ftd.predict_proba(X_test_new)

mlp_meta_submission = pd.DataFrame(np.zeros((lgbm_sub.shape[0],1)),columns = ['TARGET'])
mlp_meta_submission['TARGET'] = mlp_meta_sub[:,1]
mlp_meta_ftd.score(X_test_new,y_test)

## 메타모델 LR
meta_lr = LogisticRegression(max_iter= 5000)
meta_lr_ftd = meta_lr.fit(X_train_new,y_train)
meta_lr_sub = meta_lr_ftd.predict_proba(X_test_new)
meta_lr_ftd.score(X_test_new,y_test)

## 빅콘 데이터 예측 시작
import matplotlib.font_manager

plt.rcParams['font.family'] = 'Malgun Gothic'
font_list = matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
[matplotlib.font_manager.FontProperties(fname=font).get_name() for font in font_list if 'Hancom' in font]

import matplotlib as mpl
mpl.rcParams['font.family'] = 'Hancom Gothic'


from sklearn.metrics import confusion_matrix
## roc커브, best threshold 계산 

test_score = ftd_model.predict_proba(X_test_new)[:,1]
fpr,tpr,thresholds = roc_curve(y_test,test_score)

## roc_auc : lgbm 메타모델, 0.6422696801015308
roc_auc_score(y_test,test_score)
roc_auc_score(y_test,X_test_new[0]) #mlp
roc_auc_score(y_test,X_test_new[1]) #rf
roc_auc_score(y_test,X_test_new[2]) #et
roc_auc_score(y_test,X_test_new[3]) #lr
roc_auc_score(y_test,X_test_new[4]) #lgbm
roc_auc_score(y_test,X_test_new[5]) #ada

J = tpr - fpr
ix = np.argmax(J)
best_thresh = thresholds[ix]

## best threshold 기반으로 confusion_matrix 생성
test_score_bi = Binarizer(threshold = best_thresh).fit_transform(test_score.reshape(-1,1))

con_mat = confusion_matrix(y_test,test_score_bi)
con_mat = pd.DataFrame(con_mat,columns = ['Predcited_neg','Predcited_pos'],index = ['True_neg','True_pos'])
print('\n\n\n\n',con_mat,'\n\n\n\n\n')

plt.figure(figsize=(10,10))
plt.plot([0,1],[0,1],label='STR')
plt.plot(fpr,tpr,label='ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()


bc_score = ftd_model.predict_proba(X_bc_new)[:,1]
bc_score_bi = Binarizer(threshold = 0.2).fit_transform(bc_score.reshape(-1,1))
max(credit_score)
min(credit_score)

## 산점도 행렬
bc_score_scale = bc_score * 100
credit_score_scale = np.array(credit_score)/10

from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X = bc_score_scale.reshape(-1, 1),y = credit_score_scale.reshape(-1, 1))
credit_pred = reg.predict([[0],[100]])

plt.figure(figsize=(5,5))
plt.scatter(bc_score_scale,np.array(credit_score_scale),label='proba - credit',alpha = 0.002)
plt.plot([[0],[100]],credit_pred,'r',label = 'LinearRegression')
plt.xlabel('핀다 앱 사용자 데이터 위험도 예측(%)')
plt.ylabel('핀다 앱 신용점수')
plt.xlim(0,40)
plt.legend()
plt.grid()
plt.show()

#회귀 검정
import statsmodels.api as sm
results = sm.OLS(credit_score_scale, sm.add_constant(bc_score_scale)).fit()
results.summary()
# 피어슨 상관계수

pd.DataFrame(np.corrcoef(credit_score_scale, bc_score_scale),columns = ['신용점수','위험도'],index = ['신용점수','위험도'])


## 스태킹에 사용한 모델들의 fearture imp 그림 그리기


def make_fearture_imp_plot(model):
    imps = model.clf.feature_importances_
    try:
        f_nm = model.clf.feature_names_in_
    except:
        f_nm = model.clf.feature_name_
    data={'feature_names':f_nm ,'feature_importance':imps}
    fi_df = pd.DataFrame(data)

    #Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

    #Define size of bar plot
    plt.figure(figsize=(10,7))
    #Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    #Add chart labels
    plt.title(model.clf.__class__.__name__ + ' ' + 'FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.tight_layout()
    plt.savefig('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\ppt용 그림\\' + model.clf.__class__.__name__ +'.png')
    plt.show()
    

make_fearture_imp_plot(rf)
make_fearture_imp_plot(et)
make_fearture_imp_plot(lgbm)
make_fearture_imp_plot(ada)

# for feature

import random
sample_idx = random.sample(range(X_train.shape[0]),5000)

def shap_plot_saver(model_,X_):
    model_nm =  model_.clf.__class__.__name__
    model = model_.clf
    explainer = shap.Explainer(model.predict_proba, X_)
    shap_values = explainer(X_)
    try:
        shap.plots.beeswarm(shap_values,order=shap_values.abs.max(0),show = False,max_display = 20)
    except: 
        try:
            shap.plots.beeswarm(shap_values[:,:,1],order = shap_values.abs.max(0)[:,0],show = False,max_display = 20)
        except:
            shap.plots.beeswarm(shap_values[:,:,1],show = False,max_display = 20)
    plt.tight_layout()
    plt.savefig('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\ppt용 그림\\' + model_nm + 'shap'+'.png')
    plt.clf()
    return shap_values

X_sample = X_train.iloc[sample_idx]

MLP_value = shap_plot_saver(MLP,X_sample)
lr_value = shap_plot_saver(lr,X_sample)
lgbm_value = shap_plot_saver(lgbm,X_train)
MLP_value.abs.max(0)[:,1].shape
lgbm_value.abs.max(0).shape

rf_value = shap_plot_saver(rf,X_sample)
et_value = shap_plot_saver(et,X_sample)
ada_value = shap_plot_saver(ada,X_sample)


