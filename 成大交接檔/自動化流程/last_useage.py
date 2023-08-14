#!/usr/bin/env python
# coding: utf-8

# In[1]:


import xgboost as xgb
from sklearn import svm
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
from scipy.spatial import distance_matrix
import math


# 每週資料加總

# In[2]:


def week_data_sum(data,year):
#     刪除第53週
    data = data.drop(data[data['週'] == 53].index)
    part_compute = pd.DataFrame()
    total_sum = 0
    for num in range(1, 53,1):
        temp = data[data['週'] == num]
        last_temp = temp.tail(1)
        if len(temp) == 0:
            last_temp =  data.tail(1)
            last_temp['數量'] = 0
            last_temp['週'] = num
            last_temp['年'] = year
            print("沒有:",last_temp)
            part_compute = part_compute.append(last_temp)
        else:
            temp = temp.drop(temp[temp['數量'] < 0].index)
            last_temp['數量'] =  temp['數量'].sum()
            part_compute = part_compute.append(last_temp)
        total_sum = total_sum +part_compute['數量'].values[0]
    print(part_compute)
    print(total_sum)
   
    
    return part_compute


# In[3]:


def new_year(data):
    date = pd.DataFrame()
    new_year = [[2019,2,2,2019,2,10],[2020,1,23,2020,1,29],[2021,2,10,2021,2,16],[2022,1,31,2022,2,6]]
    for i in range(len(new_year)):
        date = pd.DataFrame()
        date['日期'] = pd.DataFrame(pd.date_range(f'{new_year[i][1]}/{new_year[i][2]}/{new_year[i][0]}',f'{new_year[i][4]}/{new_year[i][5]}/{new_year[i][3]}'))
        date['週'] = date['日期'].dt.isocalendar().week
        date.drop_duplicates(subset='週', keep='last', inplace=True)
       
    #     選出該週並判斷該週的休假日是否大於等於4
        for week in range(len(date)):
            target = data[(data["年"] == new_year[i][0]) & (data["週"] == date['週'].values[week])]
            target_index = data[(data["年"] == new_year[i][0]) & (data["週"] == date['週'].values[week])].index.astype(int)
#             如果沒有那週的資料則用去年的資料補上並做更改
            if target.empty:
                temp = data[(data["年"] == new_year[i][0]-1) & (data["週"] == date['週'].values[week])]
                temp['年'] = 2022
                temp['數量'] = 0
                temp['休假天數'] = 7
                temp['工作天數'] = 0
                data = data.append(temp,ignore_index=True)
                target = data[(data["年"] == new_year[i][0]) & (data["週"] == date['週'].values[week])]
                target_index = data[(data["年"] == new_year[i][0]) & (data["週"] == date['週'].values[week])].index.astype(int)

#                 找出2019年第五周的資料並將數量改成0，年改成2022
                
            if target['休假天數'].values[0] >= 4:
                data.loc[target_index.values[0]:target_index.values[0],'過年'] = True
#         找出有過年的那幾週的資料
    data = data[data.columns.drop(list(data.filter(regex='level_0')))]
    new_year_day = data[data['過年'] == True].reset_index()
#     print('-----------------------------',new_year_day)
  

#     找出同一年重複的，只留最少的
    for i in range(len(new_year)):
        if len(new_year_day[new_year_day['年'] == new_year[i][0]]) > 1:
#             找出不要的
            temp1 = data[(data['年']== new_year[i][0]) & (data['過年']==True)]
            temp1_index = temp1[temp1['數量'] == temp1['數量'].max()].index.values[0]
            data.loc[temp1_index:temp1_index,"過年"] = False
            temp2= new_year_day[new_year_day['年']== new_year[i][0]]
            temp2_index =temp2[temp2['數量'] == temp2['數量'].max()].index.values[0]
            new_year_day = new_year_day.drop(temp2_index)
    
#     新增使用量欄位
    new_year_day = data[data['過年'] == True].reset_index()
    new_year_day_index = data[data['過年'] == True].index.astype(int)
    for i in range(len(new_year_day)):
        if i == 0:
            before_day = data.loc[new_year_day_index[i]-1:new_year_day_index[i]-1,"數量"].round(decimals = 2)
        else:
            before_day = new_year_day.loc[i-1:i-1,'數量'].round(decimals = 2)
    #         data.loc[new_year_day_index[i]:new_year_day_index[i],f'使用量(過年)'] = before_day.values[0]
        new_year_day.loc[i:i,'使用量(過年)'] = before_day.values[0]
#     刪除過年資料並且另外處理計算
    data = data.drop(data[data['過年']==True].index).reset_index(drop=True)

    
    return data,new_year_day


# 2022資料加總返還最後一筆值

# In[6]:


def main(part_no,room,frequence):
#     讀取資料
    data=pd.read_csv(f"../使用量計算/有加工作日/差值測試/{room}/{part_no}.csv")
    data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
    total_data = data[data['年']==2022]
    total_data = week_data_sum(total_data,2022)
    total_data = total_data.sort_values(['年', '週'], ascending=True)
    answer = total_data.tail(frequence)['數量'].sum()
    
    return answer
    
    
    


# In[ ]:




