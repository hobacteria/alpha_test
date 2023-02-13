import pandas as pd
import numpy as np

path = 'C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bigcon_data\\'

col_nm_dic = {'SK_ID_CURR': 'application_id',
 'CODE_GENDER': 'gender',
 'DAYS_BIRTH': 'birth_year' ,#- 'insert_time',
 'DAYS_EMPLOYED': 'company_enter_month' ,#- 'insert_time', ## 빅콘 데이터에는 입사 연월이라 매칭시키려면 월단위로 조정해야 될듯
 'AMT_INCOME_TOTAL' : 'yearly_income',
 #'NAME_HOUSING_TYPE' : 'houseown_type', #matching_house_income_type,py로 범주 재구성, 원본 df 사용 안함
 #'NAME_INCOME_TYPE' : 'income_type', #위와 동일
 'AMT_CREDIT': 'desired_amount',
 "PREV_APP_CNT" : 'existing_loan_cnt', #prevapp.py로 산출함
 'PREV_AMT_CREDIT': 'existing_loan_amt' #위와 동일
 
 # pre app 데이터의 amt 크레딧
 # 기대출 수, 금액은 산출 가능.
 # 고용 형태는 모르겠음 -> 의논
 # 하우싱 타입도 가능은 할 것 같은데 전처리 따로 해야됨
 # 대부분 사용 가능! 
    }
col_nm_dic.values()
bc_df = pd.read_csv(path + 'user_spec.csv',usecols = col_nm_dic.values())
bc_cat = pd.read_csv(path + 'bigcon_income,housing_adj.csv')
bc_df[['NAME_HOUSING_TYPE','NAME_INCOME_TYPE']] = bc_cat[['houseown_type','income_type']]
bc_res = pd.read_csv(path + 'loan_result.csv',usecols=['application_id','loanapply_insert_time','is_applied'])
bc_credit = pd.read_csv(path + 'user_spec.csv',usecols = ['application_id','credit_score'])



bc_df = pd.merge(bc_df,bc_res,how = 'left',on = 'application_id')
bc_df = pd.merge(bc_df,bc_credit,how = 'left',on = 'application_id')

bc_df.info(null_counts= True)
bc_df = bc_df.dropna(axis=0)
bc_df.info(null_counts= True)
bc_df = bc_df[bc_df['is_applied'] == 1]



bc_df['loanapply_insert_time'] = pd.to_datetime(bc_df['loanapply_insert_time'])
apply_date = bc_df['loanapply_insert_time']

bc_df['company_enter_month'] = bc_df['company_enter_month'].astype('str').apply(lambda x: x.replace('-','')[:6])
bc_df['birth_year'] = bc_df['birth_year'].astype('str').apply(lambda x: x.replace('-','')[:4])

bc_df['company_enter_month'] = pd.to_datetime(bc_df['company_enter_month'].astype('str'),format = '%Y%m')
bc_df['birth_year'] = pd.to_datetime(bc_df['birth_year'].astype('str'),format = '%Y')

company_date = bc_df['company_enter_month']
birth_date = bc_df['birth_year']

bc_df['AGE'] = np.trunc((((apply_date - birth_date)).dt.days)/365)
bc_df['WORK_MONTH'] = np.trunc((((apply_date - company_date)).dt.days)/30)

bc_df.sort_values(by = 'loanapply_insert_time',inplace= True)
bc_df = bc_df.drop_duplicates(subset='application_id')

for new_col, old_col in col_nm_dic.items():
    bc_df.rename(columns  = {old_col : new_col},inplace=True)

bc_df['WORK_AGE'] = round(bc_df['AGE'] - (bc_df['WORK_MONTH']/12),2)


bc_df['INCOME_CREDIT_RATE'] = bc_df['AMT_INCOME_TOTAL'] / bc_df['AMT_CREDIT']
## 소득 부채비율을 구하는 과정에서 소득이 0인경우 na가 나와서 fillna로 처리
bc_df.loc[bc_df['INCOME_CREDIT_RATE'] == np.inf,['INCOME_CREDIT_RATE']] = 0
bc_df.fillna(0,inplace=True)
bc_df['PRE_CURR_CREDIT_DIFF'] = bc_df['AMT_CREDIT'] - bc_df['PREV_AMT_CREDIT']
bc_df.loc[bc_df['CODE_GENDER'] == 1,'CODE_GENDER'] = 'M'
bc_df.loc[bc_df['CODE_GENDER'] == 0,'CODE_GENDER'] = 'F'
bc_df.info()
bc_df.drop(['SK_ID_CURR','loanapply_insert_time','is_applied','DAYS_BIRTH','DAYS_EMPLOYED'],inplace=True,axis =  1)
bc_df.to_csv('C:\\Users\\ghrbs\\OneDrive - inha.edu\\바탕 화면\\2022 동계\\alpha_test\\bigcon_eval.csv',index=False)
