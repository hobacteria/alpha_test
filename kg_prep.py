import os
import pandas as pd
import numpy as np
import math
import lightgbm as lgbm
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from sklearn.neural_network import MLPClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import SMOTENC
import seaborn as sns

path = 'C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\'


col_nm_dic = {'SK_ID_CURR': 'application_id',
 'CODE_GENDER': 'gender',
 'DAYS_BIRTH': 'birth_year' ,#- 'insert_time',
 'DAYS_EMPLOYED': 'company_enter_month' ,#- 'insert_time', ## 빅콘 데이터에는 입사 연월이라 매칭시키려면 월단위로 조정해야 될듯
 'AMT_INCOME_TOTAL' : 'yearly_income',
 #'NAME_HOUSING_TYPE' : 'houseown_type', matching_house_income_type,py로 범주 재구성, 원본 df 사용 안함
 #'NAME_INCOME_TYPE' : 'income_type', 위와 동일
 'AMT_CREDIT': 'desired_amount'
 #"PREV_APP_CNT" : 'existing_loan_cnt' prevapp.py로 산출함
 #'PREV_AMT_CREDIT': 'existing_loan_amt' 위와 동일
 
 # pre app 데이터의 amt 크레딧
 # 기대출 수, 금액은 산출 가능.
 # 고용 형태는 모르겠음 -> 의논
 # 하우싱 타입도 가능은 할 것 같은데 전처리 따로 해야됨
 # 대부분 사용 가능! 신용점수 못쓰는건 조금 아쉬움
}

use_cols = col_nm_dic.keys()
# kaggle 데이터 입력
main_df = pd.read_csv(path + 'application_train.csv',usecols= use_cols)
y_train =  pd.read_csv(path + 'application_train.csv',usecols= ['TARGET'])
# 기대출수, 기대출 금액 가져오기
prev_df = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\prev_data.csv')

# 소득, 주택소유 유형 범주 수정 데이터
cat_adj_df = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\kaggle_income,housing_adj.csv')

## 빅콘 데이터
X_bc = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bigcon_eval.csv')
credit_score = X_bc.pop('credit_score')

# agg train data
main_df = pd.merge(main_df,prev_df,how = 'left', on = 'SK_ID_CURR')
main_df = pd.merge(main_df,cat_adj_df,how = 'left', on = 'SK_ID_CURR')
main_df.drop('SK_ID_CURR',axis = 1,inplace=True)


## 젠더 결측치3개, 최빈값으로 처리함.
## employ 데이터 결측치 매우 많음. 테스트 데이터에도 많은 결측치가 있기 때문에 전부 삭제하는건 안됨. mice로 대체
## 테스트 데이터는 이미 분리되어 있기 때문에 따로 해주지 않음
main_df.loc[main_df['CODE_GENDER'] == 'XNA','CODE_GENDER'] = 'M'

main_df.loc[main_df['DAYS_EMPLOYED'] == 365243,'DAYS_EMPLOYED'] = np.nan

## 다음으론 무직자의 경우 결측이 아닌가?
## 소득 유형 확인





# raw 데이터 보존
X_train = main_df.copy()
X_train.info()
## test, val 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=4885, stratify = y_train)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=42,stratify = y_train)


# 범주형 변수 설정
cat_col = ['CODE_GENDER', 'NAME_INCOME_TYPE','NAME_HOUSING_TYPE']

X_train[cat_col] = X_train[cat_col].astype('category')
X_test[cat_col] = X_test[cat_col].astype('category')
X_val[cat_col] = X_val[cat_col].astype('category')
X_bc[cat_col] = X_bc[cat_col].astype('category')

## bassline에서 lgbm은 자동으로 범주형변수를 처리해주지만 이 스크립트에서는 로지스틱 회귀 등을 사용하기 때문에
## one hot encoding을 사용해 범주형 데이터를 바이너리 데이터로 저장
## 

def out_lier(cols,df = (X_train,y_train), num = 200):
    x = df[0]
    y = df[1]
    for nm in cols:
        tmp = sorted(x[nm])[-num]
        y = y[x[nm] <= tmp]
        x = x[x[nm] <= tmp]
    return x,y

X_train ,y_train= out_lier(['AMT_INCOME_TOTAL','AMT_CREDIT',"PREV_APP_CNT",'PREV_AMT_CREDIT'])

cat_train_dummy = pd.get_dummies(X_train[cat_col])
cat_test_dummy = pd.get_dummies(X_test[cat_col])
cat_val_dummy = pd.get_dummies(X_val[cat_col])
cat_bc_dummy = pd.get_dummies(X_bc[cat_col])

## 인코딩된 범주형 변수들 다시 추가
cat_train = X_train[cat_col]
cat_test = X_test[cat_col]
cat_val = X_val[cat_col]
cat_bc = X_bc[cat_col]

X_train.drop(cat_col, axis = 1, inplace = True)
X_test.drop(cat_col, axis = 1, inplace = True)
X_val.drop(cat_col, axis = 1, inplace = True)
X_bc.drop(cat_col, axis = 1, inplace = True)

X_train = pd.concat([X_train,cat_train_dummy],axis = 1)
X_test = pd.concat([X_test,cat_test_dummy],axis = 1)
X_val = pd.concat([X_val,cat_val_dummy],axis = 1)
X_bc = pd.concat([X_bc,cat_bc_dummy],axis = 1)



## 결측치 처리, 결측치 모델 학습은 train set으로만 진행
mice = IterativeImputer(random_state=4885)
mice.fit(X_train)
X_train.loc[:,:] = mice.transform(X_train)
X_test.loc[:,:] = mice.transform(X_test)
X_val.loc[:,:] = mice.transform(X_val)

## 실험. 변수 추가

def feature_cre(X_train):
    X_train['AGE'] = np.trunc(X_train['DAYS_BIRTH'] / -365)
    X_train['WORK_MONTH'] = np.trunc(X_train['DAYS_EMPLOYED'] / -30)
    X_train['INCOME_CREDIT_RATE'] = X_train['AMT_INCOME_TOTAL'] / X_train['AMT_CREDIT']
    X_train['PRE_CURR_CREDIT_DIFF'] = X_train['AMT_CREDIT'] - X_train['PREV_AMT_CREDIT']
    X_train['WORK_AGE'] = X_train['AGE'] - (X_train['WORK_MONTH']/12)
    return X_train





X_train = feature_cre(X_train)
X_test = feature_cre(X_test)
X_val = feature_cre(X_val)

X_train.info()

drop_col = ['DAYS_BIRTH','DAYS_EMPLOYED']


X_train.drop(drop_col, axis = 1, inplace = True)
X_test.drop(drop_col, axis = 1, inplace = True)
X_val.drop(drop_col, axis = 1, inplace = True)


## 로버스트 스케일링
# RobustScaler 선언 및 학습

num_col = ['AMT_INCOME_TOTAL','AMT_CREDIT','PREV_AMT_CREDIT','AGE','WORK_MONTH','INCOME_CREDIT_RATE','PRE_CURR_CREDIT_DIFF','WORK_AGE']

robustScaler = RobustScaler().fit(X_train[num_col])
# train셋 내 feature들에 대하여 robust scaling 수행
X_train.loc[:,num_col] = robustScaler.transform(X_train[num_col])

# test셋 내 feature들에 대하여 robust scaling 수행

X_val.loc[:,num_col] = robustScaler.transform(X_val[num_col])
X_test.loc[:,num_col] = robustScaler.transform(X_test[num_col])
## bc는 모집단이 다르기 때문에 따로 스케일링

robustScaler = RobustScaler().fit(X_bc[num_col])
X_bc.loc[:,num_col] = robustScaler.transform(X_bc[num_col])



X_train = X_train[X_test.columns]
X_bc = X_bc[X_train.columns]
train_set = pd.concat([X_train,y_train],axis = 1)

val_set = pd.concat([X_val,y_val],axis = 1)
test_set = pd.concat([X_test,y_test],axis = 1)
bc_set = pd.concat([X_bc,credit_score],axis = 1)

train_set.to_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\train_set.csv',index = False)

val_set.to_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\val_set.csv',index = False)
test_set.to_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\test_set.csv', index = False)
bc_set.to_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bc_set.csv', index = False)
