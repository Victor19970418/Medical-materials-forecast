import pandas as pd
import every_week
import two_week
import four_week
import last_useage


# 讀取要預測的衛材

# In[2]:


def main(room):
    # 設定病房



    # In[3]:


    check_part_nos = pd.read_csv(f"../使用量、頻率排行/病房2022使用量用頻率排序/2022_{room}.csv",encoding='utf-8')
    check_part_nos= check_part_nos[check_part_nos['紀錄週數'] >= 13]
    check_part_nos['預測值'] = 0
    check_part_nos.head(50)


    # 刪除有問題的料號

    # In[4]:


    # B03110120
    # A04300023
    # A04200038
    # A04200020
    # A00120025
    # C03613006
    # C10220101
    # C13830005
    # C13930399
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "B03110120"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "A04300023"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "A04200038"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "A04200020"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "A00120025"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "C03613006"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "C10220101"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "C13830005"].index)
    # check_part_nos = check_part_nos.drop(check_part_nos[check_part_nos['料號'] == "C13930399"].index)
    check_part_nos = check_part_nos.reset_index(drop=True)


    # 把所有需要預測的衛材都預測過一遍

    # In[5]:


    columns = ['料號','紀錄週數']
    wrong =  pd.DataFrame()
    for i in range(len(check_part_nos)):
        temp = check_part_nos.loc[i:i]

        if temp['紀錄週數'].values[0] >= 40:
            try:
                answer = every_week.main(temp['料號'].values[0],room)
            except:   
                wrong = wrong.append(temp[columns])
                continue
            print("每週預測1次")
        elif (temp['紀錄週數'].values[0] < 40) & (temp['紀錄週數'].values[0] >= 23):
            try:
                answer = two_week.main(temp['料號'].values[0],room)
            except:   
                wrong = wrong.append(temp[columns])
                continue
            print("2週預測1次")
        else:
            try:
                answer = four_week.main(temp['料號'].values[0],room)
            except:   
                wrong = wrong.append(temp[columns])
                continue
            print("1個月預測1次")
        check_part_nos.loc[i:i,'預測值'] = answer




    # 新儲存有問題的料號

    # In[7]:


    wrong = wrong.reset_index(drop=True)
    wrong.to_csv(f"./有問題的料號/{room}.csv",encoding='utf_8_sig')


    # In[8]:




    # check_part_nos.head(50)
    for i in range(len(wrong)):
        temp = wrong.loc[i:i]
        if temp['紀錄週數'].values[0] >= 40:
            answer = last_useage.main(temp['料號'].values[0],room,1)
            print(answer)
            check_part_nos.loc[check_part_nos['料號']==temp['料號'].values[0],'預測值'] = answer
        elif (temp['紀錄週數'].values[0] < 40) & (temp['紀錄週數'].values[0] >= 23):
            answer = last_useage.main(temp['料號'].values[0],room,2)
            check_part_nos.loc[check_part_nos['料號']==temp['料號'].values[0],'預測值'] = answer
        else:
            answer = last_useage.main(temp['料號'].values[0],room,4)
            check_part_nos.loc[check_part_nos['料號']==temp['料號'].values[0],'預測值'] = answer




    # In[11]:


    check_part_nos.to_csv(f"./預測結果/{room}.csv",encoding='utf_8_sig')


# In[ ]:




