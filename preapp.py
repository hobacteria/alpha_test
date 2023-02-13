import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 데이터 불러오기
tmp_curr = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\application_train.csv')
tmp_test = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\application_test.csv')
tmp_prev = pd.read_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\home-credit-default-risk\\previous_application.csv')

## 과거 앱 데이터를 SK_ID_CURR로 재정렬.
## amt는 평균으로 처리
existing_loan_cnt = tmp_prev.set_index('SK_ID_CURR').groupby('SK_ID_CURR')['SK_ID_PREV'].count().reset_index()
existing_loan_amt = tmp_prev.set_index('SK_ID_CURR').groupby('SK_ID_CURR')['AMT_CREDIT'].mean().reset_index()
existing_loan_cnt.columns = ['SK_ID_CURR', 'PREV_APP_CNT']
existing_loan_amt.columns = ['SK_ID_CURR', 'PREV_AMT_CREDIT']

existing_loan_cnt.loc[existing_loan_cnt['PREV_APP_CNT'] >1,'PREV_APP_CNT'] = 1

## 과거 앱 데이터 train 데이터에 부착
tmp_curr_adj = pd.merge(tmp_curr,existing_loan_cnt,how = 'left',on = 'SK_ID_CURR')
tmp_curr_adj = pd.merge(tmp_curr_adj,existing_loan_amt,how = 'left',on = 'SK_ID_CURR')

## 과거 앱 데이터 test 데이터에 부착
tmp_test_adj = pd.merge(tmp_test,existing_loan_cnt,how = 'left',on = 'SK_ID_CURR')
tmp_test_adj = pd.merge(tmp_test_adj,existing_loan_amt,how = 'left',on = 'SK_ID_CURR')

res_train = tmp_curr_adj[['SK_ID_CURR', 'PREV_APP_CNT' ,'PREV_AMT_CREDIT']]
res_test = tmp_test_adj[['SK_ID_CURR', 'PREV_APP_CNT' ,'PREV_AMT_CREDIT']]

res_train.fillna(0,inplace=True)
res_test.fillna(0,inplace=True)
res_train.info()
res_train.describe()
plt.hist(res_train.set_index('SK_ID_CURR')["PREV_APP_CNT"],bins = 100)
plt.show()
plt.hist(res_train.set_index('SK_ID_CURR')["PREV_AMT_CREDIT"],bins = 100)
plt.show()
res_train.to_csv('prev_data.csv',index = False)
res_test.to_csv('prev_data_test.csv',index = False)







