# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:31:14 2021

@author: SURIMWANG
"""

#%%
import cv2
import pandas as pd 
import numpy as np
import os
os.chdir('D:/MNIST/source')
from datetime import datetime, timedelta
from glob import glob
#%% 개별 데이터 to 종합 데이터로 
data_list = glob('D:/05.Kidney/data/개별데이터/*.xlsx')
col = pd.read_excel(data_list[0]).columns
df = pd.DataFrame(columns=col)
for i in range(len(data_list)):
    one_table = pd.read_excel(data_list[i])
    df = df.append(one_table)

df.to_csv('../data/eda_data.csv', encoding = 'utf-8-sig')    
#%% 
df = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df = df.drop_duplicates()
df = df.sort_values(by=['검사명', '검체'], ascending=[True, True], axis= 0)

#%%
df_1 = df[['검체', '검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
df_1 = df_1.drop_duplicates()
df_1 = df_1.sort_values(by=['검사명'], ascending=[True], axis= 0)

df_2 = df[['검체검사결과','검사명','최저참고치', '최고참고치', '단위']].drop_duplicates()
#df_2['judge'] =1
df_2 = df_2.drop_duplicates()
df_2 = df_2.sort_values(by=['검사명'], ascending=[True], axis= 0)

test = pd.merge(df_1,df_2, how='left', on=['검체검사결과','검사명','최저참고치', '최고참고치', '단위'])

#%% 잘못된 데이터 한번에 수정하기
df.to_csv('../data/eda_data.csv', encoding = 'utf-8-sig')
for i in range(13,55):
    one_table = pd.read_excel(data_list[i])
    one_table.loc[one_table['검사명'] == 'Micro Alb Ratio', '단위'] = 'ug/mg'
    one_table.to_excel(data_list[i], encoding = 'utf-8-sig', index= False)    
    
#%%

