# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 22:37:49 2021

@author: SURIMWANG
"""

#%%
import cv2
import pandas as pd 
import numpy as np
import os
os.chdir('D:/MNIST/source')
from datetime import datetime, timedelta

#%% 저장된 파일 읽어 오기
data = pd.read_excel('../data/20210611_총데이터.xlsx', encoding = 'utf-8')
#%%1정규화 
#성별 나이 컬럼 나누기
data['성별'] = '남' 
birthday = datetime(1989, 5, 4, 0, 0, 0, 0)
age_list = []
for i in range(len(data)):
    days = data['접수일'][i] - birthday 
    age = days/timedelta(days=365)
    age = np.ceil(age)
    age_list.append(age)
data['나이'] = age_list

#'성별/나이' 컬럼 제거
data = data.drop('성별/나이', axis=1)

#%% 중복을 제거해서 컬럼별 관계를 살펴보자.
# 관계를 살펴볼 데이터만 불러오기
data_relation = data[['검사명', '최저참고치', '최고참고치', '단위', '검체검사결과', '검체']]
duplicated_data = data_relation.drop_duplicates()
duplicated_data.to_csv('../data/duplicated_data.csv')

# 관계를 살펴볼 데이터만 불러오기 2
data_relation2 = data[['검사명', '검체검사결과', '검체']]
duplicated_data2 = data_relation2.drop_duplicates()
duplicated_data2.to_csv('../data/duplicated_data2.csv')

# 검체와 검사자 관계 알아보기 
data_relation3 = data[['검체', '검사자']]
duplicated_data3 = data_relation3.drop_duplicates()
duplicated_data3.to_csv('../data/duplicated_data3.csv')


# 검체와 검사자 관계 알아보기 
data_relation4 = data[['검체', '검사자', '검체검사결과']]
duplicated_data4 = data_relation4.drop_duplicates()
duplicated_data4.to_csv('../data/duplicated_data4.csv')


# 병원과 의사 관계 알아보기 
data_relation5 = data[['병원명', '의뢰의사']]
duplicated_data5 = data_relation5.drop_duplicates()
duplicated_data5.to_csv('../data/duplicated_data5.csv') # 병원명 잘못 기입한거 있음
