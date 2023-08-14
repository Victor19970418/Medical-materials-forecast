import pandas as pd
import 高頻率更新




def main(room):
    check_part_nos = pd.read_csv(f"../使用量、頻率排行/病房2022使用量用頻率排序/2022_{room}.csv",encoding='utf-8')
    # 只有大於40周以上的需要更新
    check_part_nos= check_part_nos[check_part_nos['紀錄週數'] >= 40]



    answer = pd.DataFrame()
    columns = ['料號','紀錄週數']
    wrong =  pd.DataFrame()
    for i in range(len(check_part_nos)):
        temp = check_part_nos.loc[i:i]
        try:
            best_dataframe,basic_input = 高頻率更新.renew_column(temp['料號'].values[0],room)

            data = {'part_no':temp["料號"],'最佳組合':basic_input}
            answer = answer.append(data,ignore_index=True)
        except:   
            wrong = wrong.append(temp[columns])
            continue


    # In[5]:






