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


# 正規化

# In[2]:


features = ['週','工作天數','休假天數','前1週','前2週','前3週','前4週','每月第幾週','滑動平均',
            '總下雨量','下雨天數','沒下雨天數']

# need_normalize = ['週','工作天數','休假天數','前1週','前2週','前3週','前4週','數量','每月第幾週','總病人數','平均氣溫','前一週總病人數',
#                   '前一週平均病人數','相似日','滑動平均','流感門診人數','流感急診人數','流感總人數','年','預測病人數量','平均濕度','隔週平均溫度差']
need_normalize = ['週','工作天數','休假天數','前1週','前2週','前3週','前4週','數量','每月第幾週','相似日','滑動平均',
                  '年','最大溫差','最大平均溫度差','相似資料','前1週人數','前1週最多人', '前1週平均人數','前1週開刀人數','前1週開刀平均人數','前1週開刀最多人']
robust_need_normalize = ['週','前1週','前2週','前3週','前4週','數量','每月第幾週','平均氣溫','相似日','滑動平均','流感門診人數','流感急診人數','流感總人數']
# need_normalize = ['週','年','工作天數','休假天數','數量','前1週','前2週','前3週','前4週']


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
    
    return answer


# 最大最小值正規化

# In[4]:


#正規化
def data_normalize(total_data, need_normalize):
    data = total_data.copy()
    train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]
#     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2021)] 
#     train_data =  total_data[total_data['分類年']== 2019] 
  
#     train_data = data[data['年']== 2019] 
    #使用最大最小值進行標準化
    for i in range(len(need_normalize)):
        column = need_normalize[i]
        molecular = data[column]-train_data[column].min()
        denominator = train_data[column].max()-train_data[column].min()
        data[column] = (molecular/denominator)
    return data


# Robust Scaling正規化

# In[5]:


def robust_normalize(total_data, need_normalize):
    data = total_data.copy()
#     train_data = data[ (data['年']== 2019)|(data['年']== 2020)]     
    train_data = data[data['年']== 2019] 
        #使用最大最小值進行標準化
    for i in range(len(need_normalize)):
        column = need_normalize[i]
#         中位數
        median = train_data[column].quantile(0.5)
#         第1位數
        first_quartile = train_data[column].quantile(0.25)
#         第3位數
        third_quartile = train_data[column].quantile(0.75)
#         分子
        molecular = data[column]-median
#         分母
        denominator = third_quartile - first_quartile
        data[column] = (molecular/denominator)
    return data


# z-score正規化 

# In[6]:


def zscore_normalize(total_data, need_normalize):
    data = total_data.copy()
#     train_data = data[ (data['年']== 2019)|(data['年']== 2020)]     
    train_data = data[data['年']== 2019] 
        #使用最大最小值進行標準化
    for i in range(len(need_normalize)):
        column = need_normalize[i]
#         分子
        molecular = data[column]-train_data[column].mean()
#         分母
        denominator = train_data[column].std()
        data[column] = (molecular/denominator)
    return data


# 每週資料加總

# In[7]:


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


# 提取前n天的資料

# In[8]:


# 總資料data
# 要提取前幾天資料n_day
def take_data(data,n_day):
    data_copy = data.copy()
    # 總共要提取幾次
    for num in range(n_day, len(data_copy),1):
#         每次提取n筆
        for time in range(1,n_day+1,1):
            before_day = data_copy.loc[num-time:num-time,"數量"].round(decimals = 2)
            print(type(before_day))
            data_copy.loc[num:num,f'前{time}週'] = before_day.values[0]

    return data_copy


# 判斷第幾週

# In[9]:


def detect_week(week):
    if week == 0:
        return '第4週'
    elif week == 1:
        return '第1週'
    elif week == 2:
        return '第2週'
    else:
        return '第3週'


# 新增降水量資料&是否有下雨

# In[10]:


def weather_data(data):
    # 讀取降水資料
    precipitation = pd.read_csv(f"../使用量計算/天氣資料/斗六降水量資料1.csv")
    #     為了更使用量資料合併更改欄位名稱
    precipitation.rename(columns={'日期': '帳務日期'}, inplace=True)
    precipitation['帳務日期'] =  pd.to_datetime(precipitation['帳務日期'])
    data['帳務日期'] =  pd.to_datetime(data['帳務日期'])
    print(precipitation.head())
    print("合併前長度:",len(data))
    columns = ["年","週"]
    precipitation.drop_duplicates(subset=columns, keep='last', inplace=True)
    data = pd.merge(data, precipitation, on=columns)
    print("合併後長度:",len(data))
    data = data[data.columns.drop(list(data.filter(regex='週_y')))]
    data = data[data.columns.drop(list(data.filter(regex='年_y')))]
    data.rename(columns={'週_x': '週'}, inplace=True)
    data.rename(columns={'年_x': '年'}, inplace=True)
    data.rename(columns={'帳務日期_x': '帳務日期'}, inplace=True)
    return data
    #     return precipitation


# 整理每周開刀人數、平均開刀人數

# In[11]:


def surgery_data(data):
    people = pd.read_csv(f"../使用量計算/醫院提供病房人數資料/五病房開刀人數.csv")
    people = people[people.columns.drop(list(people.filter(regex='Unnamed')))]
    people.rename(columns={'手術日期': '帳務日期'}, inplace=True)
    people['帳務日期'] =  pd.to_datetime(people['帳務日期'])
    people['週'] = people['帳務日期'].dt.isocalendar().week
    people['年'] = people['帳務日期'].dt.isocalendar().year
    columns = ['週','年']
    print("合併前長度:",len(data))
    data = pd.merge(data, people, on=columns)
    print("合併後長度:",len(data))
    print(data)
    data.rename(columns={'週_x': '週'}, inplace=True)
    data.rename(columns={'年_x': '年'}, inplace=True)
    data.rename(columns={'帳務日期_x': '帳務日期'}, inplace=True)
    return data


# 整理每周總病人人數、平均病人人數

# In[12]:


def people_data(data):
    people = pd.read_csv(f"../使用量計算/醫院提供病房人數資料/五病房人數.csv")
    people = people[people.columns.drop(list(people.filter(regex='Unnamed')))]
    people.rename(columns={'住院日期': '帳務日期'}, inplace=True)
    people['帳務日期'] =  pd.to_datetime(people['帳務日期'])
    people['週'] = people['帳務日期'].dt.isocalendar().week
    people['年'] = people['帳務日期'].dt.isocalendar().year
    columns = ['週','年']
    print("合併前長度:",len(data))
    data = pd.merge(data, people, on=columns)
    print("合併後長度:",len(data))
    print(data)
    data.rename(columns={'週_x': '週'}, inplace=True)
    data.rename(columns={'年_x': '年'}, inplace=True)
    data.rename(columns={'帳務日期_x': '帳務日期'}, inplace=True)
    return data
    


# 計算相關性，並返回相關衛材的前一個禮拜使用量

# In[13]:


def count_corr(part_no,take_num):
    corr_data = pd.DataFrame()
    final = pd.DataFrame()

    for filename in os.listdir(f"../使用量計算/衛材相關性計算/五病房/"):
        if filename=='.ipynb_checkpoints':
            continue
        temp = pd.read_csv(f'../使用量計算/衛材相關性計算/五病房/{filename}')
        name = temp['料號'].values[0] 
        temp[name] = temp['數量']
        corr_data = pd.concat([corr_data, temp[name] ], axis = 1)
   
    print("全部資料",corr_data)
    corr = corr_data.corr()
    print("相關數值",corr[part_no])
#     排序取出前3筆最大值
    data_sort=corr[part_no].sort_values(ascending=False)
    corr_part_no = data_sort.head(take_num+1).index
#     對3個衛材進行前一週使用量提取
    for num in range(1,take_num+1,1):
        temp = pd.read_csv(f'../使用量計算/衛材相關性計算/五病房/{corr_part_no[num]}.csv')
#         提取相關的衛材前一週資料
        before_temp = take_data(temp,1)

        before_temp[corr_part_no[num]] = before_temp['前1週']
        final = pd.concat([final, before_temp[corr_part_no[num]]], axis = 1)
#     print("final長度:",len(final))
    return final


# 數據平滑

# In[14]:


def smooth(data,smooth_num):
    data['前1週'] = signal.savgol_filter(data['前1週'], len(data), smooth_num )
    data['前2週'] = signal.savgol_filter(data['前2週'], len(data), smooth_num )
    data['前3週'] = signal.savgol_filter(data['前3週'], len(data), smooth_num )
    data['前4週'] = signal.savgol_filter(data['前4週'], len(data), smooth_num )
    
    return data


# 提取工作日休假日相似日的

# In[15]:


def similar_day(data):
#     第一筆不會有相似日，因此從1開始
    for i in range(1,len(data),1):
#         找到index前幾筆資料
        temp = data.iloc[0:i]
#       提取查詢日的工作天數、休假天數
        work_day = data.loc[i:i,"工作天數"].values[0]
        qk_day = data.loc[i:i,"休假天數"].values[0]
#      目標日
        target = temp[(temp["工作天數"] == work_day) & (temp["休假天數"] == qk_day)]
#         如果找不到相似日則以前一週使用量代替
        if len(target) == 0:
             data.loc[i:i,"相似日"] = data.loc[i-1:i-1,'數量'].values[0]
        else:
             data.loc[i:i,"相似日"] = target.iloc[-1]['數量']
    return data


# 判斷過年以及替換值

# In[16]:


# 2019年過年 2月2日~2月10日
# 2020年過年 1月23日~1月29日
# 2021年過年 2月10日~2月16日
# 2022年過年 1月31日~2月6日
# 2021年2月20日雖然為假日但為補班日
# 先計算上述時間是第幾週，並判斷該週休假日是否大於等於4天
# 6、5、6、5週
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


# 計算前N筆的平均值

# In[17]:


def before_n_mean(data,window_size):
    total_data = data.reset_index(drop=True).copy()
    # 計算滑動窗口
    for num in range(window_size, len(total_data),1):
        print(num)
    #     print(num-window_size)
        average = round(total_data.loc[num-window_size:num-1,"數量"].mean(),2)
    #     print(average.round(decimals = 2))
#         standard_deviation = total_data.loc[num-window_size:num-1,"數量"].std().round(decimals = 2)
#         variation = total_data.loc[num-window_size:num-1,"數量"].var().round(decimals = 2)
        total_data.loc[num:num,'滑動平均'] = average
#         total_data.loc[num:num,'滑動標準差'] = standard_deviation
#         total_data.loc[num:num,'滑動變異數'] = variation
    return total_data


# 計算特徵的分數

# In[18]:


def feature_point(features,x_train,target,n):
    columns = []
    selector = SelectKBest(f_regression,k=len(features))
    selector.fit(x_train[features],target)
#     將計算後的P值轉為分數
    scores = -np.log10(selector.pvalues_)
#     scores = selector.pvalues_
#     印出答案&降冪排序
    print("重要特徵排序:")
    indices = np.argsort(scores)[::-1]
    for f in range(len(scores)):
        print("%0.2f %s" % (scores[indices[f]],features[indices[f]]))
#     回傳前n比當作columns
    for f in range(0,n,1):
       columns.append(features[indices[f]])
    return columns


# K-means分類

# In[19]:


def k_mean(total_data):
    k_mean = pd.DataFrame()
        # k = 1~9 做9次kmeans, 並將每次結果的inertia收集在一個list裡
    kmeans_list = [KMeans(n_clusters=k, random_state=46).fit(total_data)
                    for k in range(1, 13,1)]
    silhouette_scores = [silhouette_score(total_data, model.labels_)
                     for model in kmeans_list[1:]]
    print("輪廓係數",silhouette_scores)
#     擬合好模型後我們可以計輪廓係數，用來評估集群的成效，其 silhouette_scores 越大代表越好。
#     找出最大值的索引位置
    index = silhouette_scores.index(max(silhouette_scores))
    index = index + 2
    print(index)

    temp=KMeans(n_clusters=index, random_state=46).fit_predict(total_data)
    print(temp)

    k_mean["資料分類"] = temp+1
    return k_mean


# 流感人數計算

# In[20]:


def theflu_sum(total_data):
    concat_data = pd.DataFrame()
    clinic=pd.read_csv(f"../爬蟲/流感門診人數.csv")
    emergency=pd.read_csv(f"../爬蟲/流感急診人數.csv")
    clinic = clinic.rename(columns={"就診人次": "流感門診人數"})
    emergency = emergency.rename(columns={"就診人次": "流感急診人數"})
    columns = ['年','週']
    concat_data = pd.merge(clinic, emergency, on=columns)
    concat_data = concat_data[['年','週','流感門診人數','流感急診人數']]
    concat_data['流感總人數'] = concat_data[['流感門診人數','流感急診人數']].sum(1)
    total_data = pd.merge(total_data, concat_data, on=columns)
#     print("就診人數",total_data.head())
#     print(total_data.columns)
    return total_data


# 合併預測的病房人數

# In[21]:


def pred_people(total_data):
    people=pd.read_csv(f"../預測病人人數/arima五病房人數預測.csv")
    columns = ['年','週']
    concat_data = pd.merge(total_data, people, on=columns)
    print(len(total_data))
    print("合併後資料",concat_data.head(100))
    return concat_data


# SMOGN擴增資料集

# In[22]:


def SMOGN(total_train):
    print("擴增前:",total_train)
    train_smogn = smogn.smoter(

         ## main arguments
        data = total_train,           ## pandas dataframe
        y = '數量',          ## string ('header name')
        k = 7,                    ## positive integer (k < n)
        samp_method = 'extreme',  ## string ('balance' or 'extreme')
#         random_seed = 1,
      
        ## phi relevance arguments
        rel_thres = 0.80,         ## positive real number (0 < R < 1)
        rel_method = 'auto',      ## string ('auto' or 'manual')
        rel_xtrm_type = 'high',   ## string ('low' or 'both' or 'high')
        rel_coef = 0.5           ## positive real number (0 < R)
        
    )
#     print("擴增後的資料:",train_smogn)
    return train_smogn


# 計算平均溫度差

# In[23]:


def Temperature_difference(total_data):

    for i in range(len(total_data)):
        if i != 0: 
            temp = total_data.loc[i:i,'平均氣溫'].values[0]
            pre_temp = total_data.loc[i-1:i-1,'平均氣溫'].values[0]
            total_data.loc[i:i,"隔週平均溫度差"] = abs(temp - pre_temp)
            
        else:
            total_data.loc[i:i,"隔週平均溫度差"] = 0
    return total_data


# 每一週最大溫差

# In[24]:


def MaxTemperature_difference(data):
    total_data = pd.DataFrame()
#     讀取溫度差資料
    Temperature=pd.read_csv(f"../使用量計算/天氣資料/2022溫度差資料.csv")
#     刪除第53週
#     刪除第53週
    Temperature = Temperature.drop(Temperature[Temperature['週'] == 53].index)
#     過濾資料，只保留每週最大值
    Temperature = Temperature.sort_values('最大溫差', ascending=False).drop_duplicates(subset=['年', '週'], keep='first')
    Temperature = Temperature.sort_values(['年', '週'], ascending=True)
#     刪除不必要的欄位
    Temperature = Temperature[Temperature.columns.drop(list(Temperature.filter(regex='Unnamed')))].reset_index(drop=True)
#     合併資料
    columns = ['年', '週']
    total_data = pd.merge(data, Temperature, on=columns)
    print("溫差",total_data)
    return total_data


# In[25]:


def MAPE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def nMAE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))/y_true.mean() * 100

def RMSE(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.sqrt(((y_pred - y_true) ** 2).mean())


# In[26]:


def put_data_back(y_test,y_pred,new_year_day,test_year):

    y_test_copy = y_test.copy()
    y_pred = pd.DataFrame(y_pred,columns=['數量'])
    y_pred_copy = y_pred.copy()
#     算出他維資料集中的第幾筆
    new_year_day = new_year_day[new_year_day['年'] == test_year]
    y_test_copy = y_test_copy.reset_index()
    y_test_copy = y_test_copy[y_test_copy.columns.drop(list(y_test_copy.filter(regex='index')))]
#     插入被刪除過年資料(y_test)
    df_part_1 = y_test_copy.loc[0:3]
    df_part_2 = y_test_copy.loc[4:]
#     print(df_part_2)
    y_test = df_part_1['數量'].append(new_year_day['數量'],ignore_index = True)
    y_test = y_test.append(df_part_2['數量'],ignore_index = True)
  #     插入被刪除過年資料(y_pred)
    df_part_1_pred = y_pred_copy.loc[0:3]
    df_part_2_pred = y_pred_copy.loc[4:]
    
    y_pred = df_part_1_pred['數量'].append(new_year_day['使用量(過年)'],ignore_index = True)
    y_pred = y_pred.append(df_part_2_pred['數量'],ignore_index = True)
    return y_test,y_pred


# 3個標準差異常排除

# In[27]:


def zscore_outlier(total_data,train_data):
    mean = train_data["數量"].mean()
    std = train_data["數量"].std()
#     if std < 1:
#         Threshold = 9
#     else:
#         Threshold = 3
    Threshold = 2
#     移除前長度
    before_len = len(total_data)
    total_data['zscore'] = ( total_data["數量"] - mean ) / std
    total_data = total_data[(total_data['zscore']<Threshold) & (total_data['zscore']>(-1*Threshold))]
#     移除後長度
    after_len = len(total_data)
  
    print("平均值",mean)
    print("標準差",std)
    print("長度",len(total_data))
    
    return total_data,before_len-after_len


# 1.5倍4分衛距異常值移除

# In[28]:


def Iqr_outlier(total_data,train_data):
    data = total_data.copy()
    
    
    re_before = len(data)
    n=1.5
    Q3 = np.percentile(train_data['數量'],75) 
    Q1 = np.percentile(train_data['數量'],25)
    #IQR = Q3-Q1
    IQR = Q3 - Q1 
    
    #outlier step
    outlier_step = n * IQR
    dq3 = data[~(data['數量'] < Q3 + outlier_step)]
    dq1 = data[~(data['數量'] > Q1 - outlier_step)]
    
    outlier = pd.concat([dq3, dq1])


    #outlier = Q3 + n*IQR 
    data=data[data['數量'] < Q3 + outlier_step]
    #outlier = Q1 - n*IQR 
    data=data[data['數量'] > Q1 - outlier_step]

    
    re_after = len(data)

    print(f'移除前: {re_before}, 移除後: {re_after}, 共移除 {re_before-re_after} 筆') 
    print(f'IQR: {IQR}, Q3: {Q3}, Q1: {Q1}, outlier(1.5*IQR): {outlier_step}, Q3+outlier: { Q3 + outlier_step}, Q1-outlier: {Q1 - outlier_step}')
    remove_len = re_before-re_after
    
#     return data, outlier
    return data,remove_len


# 算出相似資料的平均值算四分衛距

# In[29]:


def average_count_distance(data):
    total_data = data.copy()
    temp = total_data[0:154]
    temp = temp[['前1週','前2週','前3週','工作天數','最大溫差']]
    find_value = pd.DataFrame(distance_matrix(temp.values, temp.values))
#     拉出第一行刪除0
    first_line  = find_value[0]
    first_line = first_line[first_line != 0 ]
    answer = np.percentile(first_line,75) 
#     print("第3衛距",answer)
#     answer = find_value[0].sum()/154
    return answer


# 找相似資料

# In[30]:


# 需要先找全部距離平均值
def count_distance(data):
    averange = average_count_distance(data)
#     尋找之前跟自己距離最近的資料
    total_data = data.copy()
    total_data = total_data.reset_index(drop=True)
    for i in range(len(total_data)):
        temp = total_data[0:i+1]
        temp = temp[['前1週','前2週','前3週','工作天數','最大溫差']]
#         第一筆不找相近的值
        if i == 0:
            answer = 0
#       temp = total_data.loc[i:i,'平均氣溫'].values[0]
#             pre_temp = total_data.loc[i-1:i-1,'平均氣溫'].values[0]
        else:
            find_value = pd.DataFrame(distance_matrix(temp.values, temp.values))
#             不要選到0所以把0改成100000
            find_value = find_value.replace(0,100000)
         
#             設定一個閥值，如果沒有大於閥值則使用前一個禮拜，如果有就換成相似資料
            find_value_index = find_value.idxmin(axis = 1, skipna = True)[i]
            print("各筆資料:",find_value[i][find_value_index])
            if find_value[i][find_value_index] > averange:
                answer = temp.tail(1)['前1週'].values[0]
                print("沒有距離小於50")
                print(answer)
            else:
                print("有距離小於40")
                answer = total_data.loc[find_value_index:find_value_index,'數量'].values[0]
    
        total_data.loc[i:i,'相似資料'] = answer
        
    return total_data
    


# 進行差值

# In[31]:


def interpolation(data):

#     1.先判斷是否有0的資料，將0的資料改成nan np.nan
    data.loc[data['數量']==0,'數量'] = np.nan
#     #     如果第一筆是nan改成0
#     first_part_no = data.loc[0:0]
#     if first_part_no['數量'].values[0] == np.nan:
#         data.loc[0:0,'數量'] = 0
#     2.利用插值套件進行線性插值，插值套件會將nan的值進行插值
    data['數量'] = data['數量'].interpolate(method = 'polynomial',order=2).round(decimals = 2)
    print("更改數值:",len(data) )


    
    return data
    


# 主程式

# In[32]:


def main(part_no,room):
    need_normalize = ['週','工作天數','休假天數','前1週','前2週','前3週','前4週','數量','每月第幾週','相似日','滑動平均',
              '年','最大溫差','最大平均溫度差','相似資料','前1週人數','前1週最多人', '前1週平均人數','前1週開刀人數','前1週開刀平均人數','前1週開刀最多人']
      # 要提取前幾天的資料
    before_n = 4
    # 提取相關性資料筆數
    take_num = 4
    # 平滑參數
    smooth_num = 2
    # 提取前n筆的平均值(滑動平均)
    window_size = 3
    columns = ['週','每月第幾週','前1週','前2週','前3週','休假天數','最大溫差','相似資料','前1週開刀最多人']
    data=pd.read_csv(f"../使用量計算/有加工作日/差值測試/{room}/{part_no}.csv")
    data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
    # data['年'] =  pd.to_datetime(data['帳務日期']).dt.year

    data_2019 = data[data['年']==2019]
    data_2020 = data[data['年']==2020]
    data_2021 = data[data['年']==2021]
    data_2022 = data[data['年']==2022]


    # 把每一週的使用量加總
    data_2019 = week_data_sum(data_2019,2019)
    data_2019 = work_day(data_2019,2019)
    data_2020 = week_data_sum(data_2020,2020)
    data_2020 = work_day(data_2020,2020)
    data_2021 = week_data_sum(data_2021,2021)
    data_2021 = work_day(data_2021,2021)
    data_2022 = week_data_sum(data_2022,2022)
    data_2022 = work_day(data_2022,2022)



    total_data = pd.concat([data_2019, data_2020])
    total_data = pd.concat([total_data, data_2021])
    total_data = pd.concat([total_data, data_2022])
    total_data = total_data.reset_index()


    # 提取相似日
    total_data =  similar_day(total_data)

    total_data,new_year_day = new_year(total_data)
#     判斷是否為過年週如果不是並且使用量是0的情況底下，進行差值的運算
    total_data = interpolation(total_data)
    
    total_data = MaxTemperature_difference(total_data)



                                       
    # 算出前n筆的平均值
    total_data = before_n_mean(total_data,window_size)
    
    # 新增病房人數
    total_data = people_data(total_data)
#     新增開刀人數
    total_data = surgery_data(total_data)


    # 提取前幾天資料
    total_data = total_data.reset_index()
    total_data = take_data(total_data,before_n)
    total_data.drop(total_data.head(before_n).index,inplace=True) # 从头去掉n行
    total_data['帳務日期'] =  pd.to_datetime(total_data['帳務日期'])
    
    # 找出相似的資料
    total_data =  count_distance(total_data)
    print("相似的資料",total_data)
 

    # 新增該資料為每月的第幾週
    total_data['每月第幾週'] = total_data['週'] % 4
   

    # 按年、週排序
    total_data = total_data.sort_values(['年'], ascending=True).reset_index(drop=True)
    total_data = total_data.sort_values(['週'], ascending=True).reset_index(drop=True)

  


    total_data['分類年'] = total_data['年']
    # 訓練資料
    train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]  
#     print("排除後長度",len(total_data))
    train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]  


    target_min, target_max = train_data['數量'].min(), train_data['數量'].max()
   

    # need_normalize = need_normalize + corr_data_name

    # 最大最小值正規化
    total_data = data_normalize(total_data, need_normalize)
    
    # 做完正規畫後從新給予訓練資料
    train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]   
#     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2021)] 
#     train_data =  total_data[total_data['分類年']== 2019] 

    y_train = train_data["數量"]
    # columns = feature_point(features,x_train,y_train,8)
    # columns = columns + ["資料分類"]

    x_train = train_data[columns]

    #測試資料
    test_data = total_data[total_data['分類年']== 2022]     
    x_test = test_data[columns]
    y_test = test_data["數量"]
    
#     SVR模型
    SVRModel=svm.SVR(C=2, kernel="rbf", gamma='auto')

#     # 使用訓練資料訓練模型
    SVRModel.fit(x_train,y_train)
    y_pred=SVRModel.predict(x_test)
    
       # 反正規化
    y_test = y_test * (target_max - target_min) + target_min
    y_pred = y_pred * (target_max - target_min) + target_min
    y_test,y_pred = put_data_back(y_test,y_pred,new_year_day,2022)
    return math.ceil(y_pred.tail(1).values[0])


# In[33]:





# 讀取資料，切割資料集

# In[ ]:


# # B03106859
# # C13930399
# # A00120213
# # A02120180
# # A02320340
# # A02322259
# # A04411285
# # A04800036
# # B00206057 50週
# # B03110120
# # G81500272
# # K80004044
# room = "差值測試"

# total_answer_1 = pd.DataFrame()

# for filename in os.listdir(f"../使用量計算/有加工作日/{room}/"):
#     index = filename.index('.')
#     file_name = filename[:index]
#     part_no = file_name
#     print("衛材名稱:",part_no)
    
#     need_normalize = ['週','工作天數','休假天數','前1週','前2週','前3週','前4週','數量','每月第幾週','相似日','滑動平均',
#                   '年','最大溫差','最大平均溫度差','相似資料','前1週人數','前1週最多人', '前1週平均人數','前1週開刀人數','前1週開刀平均人數','前1週開刀最多人']

#     # 要提取前幾天的資料
#     before_n = 4
#     # 提取相關性資料筆數
#     take_num = 4
#     # 平滑參數
#     smooth_num = 2
#     # 提取前n筆的平均值(滑動平均)
#     window_size = 3

   
#     columns = ['週','每月第幾週','前1週','前2週','前3週','休假天數','最大溫差','相似資料','前1週開刀最多人']
#     SMOGN_columns = ['工作天數','休假天數','年','前1週','前2週','前3週','前4週','每月第幾週','平均氣溫','滑動平均','數量']

#     # columns = ['前1週','前2週','前3週','前4週','年','週']


#     data=pd.read_csv(f"../使用量計算/有加工作日/{room}/{part_no}.csv")
#     data = data[data.columns.drop(list(data.filter(regex='Unnamed')))]
#     # data['年'] =  pd.to_datetime(data['帳務日期']).dt.year

#     data_2019 = data[data['年']==2019]
#     data_2020 = data[data['年']==2020]
#     data_2021 = data[data['年']==2021]
#     data_2022 = data[data['年']==2022]


#     # 把每一週的使用量加總
#     data_2019 = week_data_sum(data_2019,2019)
#     data_2019 = work_day(data_2019,2019)
#     data_2020 = week_data_sum(data_2020,2020)
#     data_2020 = work_day(data_2020,2020)
#     data_2021 = week_data_sum(data_2021,2021)
#     data_2021 = work_day(data_2021,2021)
#     data_2022 = week_data_sum(data_2022,2022)
#     data_2022 = work_day(data_2022,2022)

#     total_data = pd.concat([data_2019, data_2020])
#     total_data = pd.concat([total_data, data_2021])
#     total_data = pd.concat([total_data, data_2022])
#     total_data = total_data.reset_index()


#     # 提取相似日
#     total_data =  similar_day(total_data)

#     total_data,new_year_day = new_year(total_data)
# #     判斷是否為過年週如果不是並且使用量是0的情況底下，進行差值的運算
#     total_data = interpolation(total_data)
    
#     total_data = MaxTemperature_difference(total_data)



                                       
#     # 算出前n筆的平均值
#     total_data = before_n_mean(total_data,window_size)
    
#     # 新增病房人數
#     total_data = people_data(total_data)
# #     新增開刀人數
#     total_data = surgery_data(total_data)


#     # 提取前幾天資料
#     total_data = total_data.reset_index()
#     total_data = take_data(total_data,before_n)
#     total_data.drop(total_data.head(before_n).index,inplace=True) # 从头去掉n行
#     total_data['帳務日期'] =  pd.to_datetime(total_data['帳務日期'])
    
#     # 找出相似的資料
#     total_data =  count_distance(total_data)
#     print("相似的資料",total_data)
 

#     # 新增該資料為每月的第幾週
#     total_data['每月第幾週'] = total_data['週'] % 4
   

#     # 按年、週排序
#     total_data = total_data.sort_values(['年'], ascending=True).reset_index(drop=True)
#     total_data = total_data.sort_values(['週'], ascending=True).reset_index(drop=True)

  


#     total_data['分類年'] = total_data['年']
#     # 訓練資料
#     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]  
# #     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2021)] 
# #     train_data =  total_data[total_data['分類年']== 2019] 
# #         異常值排除
# #     print("排除前長度",len(total_data))
# #     total_data,remove_score = Iqr_outlier(total_data,train_data)
# #     total_data,remove_score = zscore_outlier(total_data,train_data)

   
# #     print("排除後長度",len(total_data))
#     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]  


#     target_min, target_max = train_data['數量'].min(), train_data['數量'].max()
   

#     # need_normalize = need_normalize + corr_data_name

#     # 最大最小值正規化
#     total_data = data_normalize(total_data, need_normalize)
    






#     # 做完正規畫後從新給予訓練資料
#     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2020)|(total_data['分類年']== 2021)]   
# #     train_data =  total_data[(total_data['分類年']== 2019)|(total_data['分類年']== 2021)] 
# #     train_data =  total_data[total_data['分類年']== 2019] 

#     y_train = train_data["數量"]
#     # columns = feature_point(features,x_train,y_train,8)
#     # columns = columns + ["資料分類"]

#     x_train = train_data[columns]

#     #測試資料
#     test_data = total_data[total_data['分類年']== 2022]     
#     x_test = test_data[columns]
#     y_test = test_data["數量"]
    
#     #         XGB模型
# #     xgbrModel = xgb.XGBRegressor(learning_rate=0.01, 
# #                         gamma = 0.01, 
# #                         max_depth=2,
# #                         colsample_bytree=0.1,
# #                         reg_lambda=0.01,
# #                         seed=1,
# #                         subsample=0.1,
# #                         min_child_weight=1,
# #                         n_estimators=668)
# #     xgbrModel.fit(x_train,y_train)
# #     y_pred=xgbrModel.predict(x_test)
# #     SVR模型
#     SVRModel=svm.SVR(C=2, kernel="rbf", gamma='auto')

# #     # 使用訓練資料訓練模型
#     SVRModel.fit(x_train,y_train)
#     y_pred=SVRModel.predict(x_test)
    
#        # 反正規化
#     y_test = y_test * (target_max - target_min) + target_min
#     y_pred = y_pred * (target_max - target_min) + target_min
#     y_test,y_pred = put_data_back(y_test,y_pred,new_year_day,2022)

# #     mape = round(MAPE(y_test, y_pred),2)
#     rmse = round(RMSE(y_test, y_pred),2)
#     mae = round(nMAE(y_test, y_pred),2)
#     pred_result = pd.DataFrame({'料號': part_no,'P(RMSE)': rmse,'P(Mae)': mae,'移除比數':remove_score},index=[0])
#     total_answer_1 = total_answer_1.append(pred_result,ignore_index=True)

