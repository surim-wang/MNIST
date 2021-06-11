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
data = pd.read_excel('../data/2020-04-10.xlsx', encoding = 'utf-8')
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
