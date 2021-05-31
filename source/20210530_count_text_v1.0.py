# -*- coding: utf-8 -*-
"""
Created on Sun May 30 18:20:30 2021

@author: SURIMWANG
"""

#%%
#라이브러리 불러오기
import pandas as pd
from pandas import Series, DataFrame
import os
os.chdir('D:/MNIST/source')

#데이터 불러오기
data = pd.read_excel('../data/example_one_column.xlsx')

#한글자씩 리스트로 만들기
Deduplication_list = []
for i in range(0,len(data)-1):
    print(list(data.iloc[i,0]))
    lst = list(data.iloc[i,0])
    Deduplication_list.append(lst)

#모든 글자 하나의 리스트로 모으기
total=[]
for i in range(0, len(Deduplication_list)):
    total += Deduplication_list[i]

#total 정렬 하기 
total.sort() 


# 중복제거한 값 살펴보기
total_set_list = list(set(total))
# 정렬하기
total_set_list.sort()


# 단어별 빈도수 구하기
# 빈도수를 기록할 데이터 프레임 만들기
data = DataFrame({'name': total_set_list,
                  'count': []*152})

# 빈도수 구하기
for i, name in enumerate(data['name']):
    print(i, name)
    cnt = total.count(name)
    data.iloc[i,1] = cnt

# 정렬해서 살펴보기
data = data.sort_values(by=['count'], axis=0, ascending = False)

data = data.reset_index(drop=True)
















