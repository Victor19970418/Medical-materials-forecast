#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
from sklearn.cross_decomposition import PLSRegression
import pandas as pd
import numpy as np
import sklearn
import os
from scipy import signal
import joblib
from sklearn.feature_selection import SelectKBest,f_regression,chi2
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import smogn
import pandas
from pmdarima.arima import ndiffs
#building the model
import six
import sys
sys.modules['sklearn.externals.six'] = six 
# import mlrose
import joblib
sys.modules['sklearn.externals.joblib'] = joblib
from statsmodels.tsa.arima_model import ARIMA
import pmdarima as pm
import math


# 每週資料加總

# In[2]:


def week_data_sum(data,year):
#     刪除第53週
    data = data.drop(data[data['週'] == 53].index)
    part_compute = pd.DataFrame()
    total_sum = 0
    for num in range(1, 53,2):
        temp_1 = data[data['週'] == num]
        temp_2 = data[data['週'] == num+1]
        last_temp = temp_1.tail(1)
        print("檢查",temp_1['休假天數'])
        vacation = temp_1['休假天數'].values[0] + temp_2['休假天數'].values[0]
        work =  temp_1['工作天數'].values[0] + temp_2['工作天數'].values[0]
        last_temp['休假天數'] = vacation
        last_temp['工作天數'] = work
        print(vacation)
        if len(temp_1) == 0 & len(temp_2) == 0:
            last_temp['數量'] = 0
            last_temp['週'] = num
            last_temp['年'] = year
            part_compute = part_compute.append(last_temp)
        else:
            temp_1 = temp_1.drop(temp_1[temp_1['數量'] < 0].index)
            temp_2 = temp_2.drop(temp_2[temp_2['數量'] < 0].index)
            last_temp['數量'] =  temp_1['數量'].sum() + temp_2['數量'].sum() 
            part_compute = part_compute.append(last_temp)
        total_sum = total_sum +part_compute['數量'].values[0]
    print(part_compute)
    print(total_sum)
    
    return part_compute


# 重新計算工作日

# In[3]:


def work_day(data,year):
#     先刪除工作日和休假日
    data = data.drop(['工作天數', '休假天數'], axis=1)
    
    check_date = pd.read_csv(f"../使用量計算/週期資料.csv",encoding='utf-8')
    answer =  pd.DataFrame()
    date =  pd.DataFrame()
    date['日期'] = pd.DataFrame(pd.date_range(f'1/1/{year}',f'31/12/{year}'))
    date['週'] = date['日期'].dt.isocalendar().week
    date['年'] = date['日期'].dt.isocalendar().year
    date = date.drop(date[date['週'] == 53].index).reset_index()
    del_data =  date[date['週'] == 52].tail(1).index
    date = date.drop(date[del_data.values[0]+1:].index).reset_index()
    date["日期"] = pd.to_datetime(date["日期"] ,format='%Y/%m/%d')
    check_date["date"] = pd.to_datetime(check_date["date"] ,format='%Y/%m/%d')
    # 如果是工作日就存成true，否則存成false
    date['休假日']=date['日期'].map(lambda x:(check_date['date']==x).any())
    print(date)

    for num in range(1, 53,1):
        temp = date[date['週'] == num]
#         計算該周工作日
        holiday = len(temp[temp['休假日'] == True])
        work_day = len(temp[temp['休假日'] == False])
        date.loc[date['週']==num,'工作天數'] = work_day
        date.loc[date['週']==num,'休假天數'] = holiday
        if year == 2021 and num == 52:   
            date.loc[date['週']==num,'休假天數'] = 2

#     同一週期工作日、休假日都一樣所以只保留一筆
    date.drop_duplicates(subset='週', keep='last', inplace=True)
    date = date.drop(['日期', '休假日','level_0','index'], axis=1)
  
#     answer = pd.merge(data, date, on='週',how='outer')

    answer = pd.merge(data, date, on='週')
#     answer = answer.sort_values(['帳務日期'], ascending=True).reset_index(drop=True)

#     刪除日期和工作日
    answer.rename(columns={'年_x': '年'}, inplace=True)
    print("查看合併",answer)
    return answer


# 資料集不足的週數補齊

# In[4]:


def Compensation(data,start_yeat,end_year):
#      保存缺失的值
    part_compute = pd.DataFrame()
    last_temp = data.head(1)
    for year in range(start_yeat, end_year+1,1):
        for num in range(1, 53,1):
            temp = data[(data['週'] == num)&(data['年'] == year)]
            if temp.empty == True:
                last_temp['數量'] = 0
                last_temp['週'] = num
                last_temp['年'] = year
                part_compute = part_compute.append(last_temp)
    total_data = pd.concat([data, part_compute])
    print("長度",len(total_data))
    
    return total_data


# 進行差值

# In[5]:


def interpolation(data):

#     1.先判斷是否有0的資料，將0的資料改成nan np.nan
    data.loc[data['數量']==0,'數量'] = np.nan
#     2.利用插值套件進行線性插值，插值套件會將nan的值進行插值
    data['數量'] = data['數量'].interpolate(method = 'polynomial', order = 2).round(decimals = 2)
    print("更改數值:",len(data) )

    return data


# Outlier異常值排除

# In[6]:


def outlier(total_data,train_data):
    mean = train_data["數量"].mean()
    std = train_data["數量"].std()
    if std < 1:
        Threshold = 9
    else:
        Threshold = 3
    total_data['zscore'] = ( total_data["數量"] - mean ) / std
    total_data = total_data[(total_data['zscore']<Threshold) & (total_data['zscore']>(-1*Threshold))]
#     算出測試資料還剩幾筆
    test_data = total_data[total_data['年']== 2022]  
    print("平均值",mean)
    print("標準差",std)
    print("長度",len(total_data))
    return total_data,26-len(test_data),104-len(total_data)


# In[7]:


part_no ='K82430329'
room = '五病房'
def main(part_no,room):
    from pmdarima.arima import  auto_arima
    part_no =part_no
    room = room
    data=pd.read_csv(f"../使用量計算/有加工作日/差值測試/{room}/{part_no}.csv")
    data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
    # data['年'] =  pd.to_datetime(data['帳務日期']).dt.year

    # 補其不足的週數
    data = Compensation(data,2019,2022)
    data_2019 = data[data['年']==2019]
    data_2019 = work_day(data_2019,2019)
    data_2020 = data[data['年']==2020]
    data_2020 = work_day(data_2020,2020)
    data_2021 = data[data['年']==2021]
    data_2021 = work_day(data_2021,2021)
    data_2022 = data[data['年']==2022]
    data_2022 = work_day(data_2022,2022)

   

    # 把每一週的使用量加總
    data_2019 = week_data_sum(data_2019,2019)
    data_2020 = week_data_sum(data_2020,2020)
    data_2021 = week_data_sum(data_2021,2021)
    data_2022 = week_data_sum(data_2022,2022)
    total_data = pd.concat([data_2019, data_2020])
    total_data = pd.concat([total_data, data_2021])
    total_data = pd.concat([total_data, data_2022])
    total_data = total_data.reset_index()

    # if (len(data_2019) < 40) & (len(data_2019) >= 23):
    #     return -1000
    # if (len(data_2020) < 40) & (len(data_2020) >= 23):
    #     return -1000
    # if (len(data_2021) < 40) & (len(data_2021) >= 23):
    #     return -1000
    #     判斷是否為過年週如果不是並且使用量是0的情況底下，進行差值的運算
    total_data = interpolation(total_data)
# 切割訓練測試集
    columns = ['數量']
    # 切割訓練資料
    train_data = total_data[(total_data['年']== 2020) | (total_data['年']== 2019) |(total_data['年']== 2021 )]     
    # train_data = data[data['年']== 2019]     

    total_data,len_outlier,len_total = outlier(total_data,train_data)
    train_data = total_data[(total_data['年']== 2020) | (total_data['年']== 2019) |(total_data['年']== 2021 )]  
    # 刪除前4筆
    # train_data = train_data.drop(train_data.head(window_size).index).reset_index(drop=True) 
    train_data = train_data[columns].reset_index(drop=True) 

    #測試資料
    test_data = total_data[total_data['年']== 2022]     
    test_data = test_data[columns].reset_index(drop=True) 

    # 算出推薦的差分次數
    diff =  ndiffs(train_data["數量"],  test="adf")
# 建立模型
    arima_model =auto_arima(train_data, d=1,test='adf',
                              max_p=52, max_d=10, max_q=52, 
                              m=8, seasonal=False, D=1,
                              error_action='warn',trace = True,
                              supress_warnings=True,stepwise = True,
                              random_state=20,n_fits = 50)
# 儲存模型
    joblib.dump(arima_model,f'./2週預測Arima模型/五病房/{part_no}.pkl')
# 儲存所有預測
    prediction = pd.DataFrame()

    # 讀取模型
    auto_arima = joblib.load(f"./2週預測Arima模型/五病房/{part_no}.pkl")
    auto_arima.fit(train_data)
    temp = pd.DataFrame(auto_arima.predict(n_periods = 1))
    prediction = prediction.append(temp)

    # 測試資料長度
    len_test = len(test_data)
    for i in range(0,25-len_outlier,1):
        print(i)
        target_week_1 = test_data.loc[i:i]

        train_data = train_data.append(target_week_1)
        train_data = train_data.drop([0], axis=0).reset_index(drop=True)  
        auto_arima.fit(train_data)
        temp = pd.DataFrame(auto_arima.predict(n_periods = 1))
        prediction = prediction.append(temp)


    prediction = prediction.reset_index(drop=True)  
    prediction.columns = ['predicted_usage']
    # 無條件進位
    y_pred = prediction['predicted_usage']
    return math.ceil(y_pred.tail(1).values[0])

