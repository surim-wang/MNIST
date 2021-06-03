# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:31:44 2021

@author: SURIMWANG
"""

#%%
import cv2
import numpy as np
import json
import os
os.chdir('D:/MNIST/source')

#%%
#json 파일 읽기전에 원본을 불러와서 비교 준비하기
image = cv2.imread('../image/SCAN_01.jpg')

#labelme로 라벨링한 json 파일 읽기
with open('../image/SCAN_01.json', "r", encoding='UTF8') as st_json:
    st_python = json.load(st_json)
#출처: https://rfriend.tistory.com/474 [R, Python 분석과 프로그래밍의 친구 (by R Friend)]

#만들어진 json파일 분석하기    
st_python
#size : 7 
#1 flags :?
#2 imageData:?
#3 ImageHeight:이미지 행의 크기
#4 imagePath:이미지 이름
#5 imageWidth:이미지 렬의 크기
#6 shapes:라벨링 값
#7 version:리벨미의 버전

#%% st_python의 6번 분석하기
st_python['shapes'][0] 
#{'label': '의', 
# 'points': [[57.57575757575757, 103.03030303030302], #사각형의 첫번째 포인트
#  [93.33333333333333, 137.57575757575756]], #사각형의 두번째 포인트
# 'group_id': None,
# 'shape_type': 'rectangle',
# 'flags': {}}
## 결론: 라벨미로 라벨링이 끝나면 파이썬에서 불러와 좌표에 맞게 이미지에 사각형을 그려주고 그 포인트 값을 한줄로 바꿔서 저장하면 
## MNIST의 데이터셋과 유사하게 저장할 수 있는데 다만! 이미지 마다 사이즈가 다르기 때문에 28*28처럼 748사이즈로 맞출 수 없다는 문제가 있다.
## 이럴 경우에는 어떻게 해결 할 수 있을까?

# 글자일 경우에는 MNIST처럼 같은 사이즈로 만든다 쳐도 동물과같은 비정형한 모형의 경우에는 라벨링을 한 데이터셋은 어떤 형태가 될까?
## 

#이미지에 사각형 그리기
label = st_python['shapes'][0]['label']
x1 = int(st_python['shapes'][0]['points'][0][0])
y1 = int(st_python['shapes'][0]['points'][0][1])

x2 = int(st_python['shapes'][0]['points'][1][0])
y2 = int(st_python['shapes'][0]['points'][1][1])
f_point = tuple([x1, y1])
l_point = tuple([x2, y2])

cv2.rectangle(image, f_point, l_point, (255,0,0), 2)

#이미지 보기
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/draw_rectangle.jpg', image) # 파일 경로 & 명, 파일

#%% 반복문으로 라벨미에서 그린 사각형 모드 표시하기
for i in range(len(st_python['shapes'])):
    label = st_python['shapes'][i]['label']
    # 첫좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    
    f_point = tuple([x1, y1])
    l_point = tuple([x2, y2])
    
    cv2.rectangle(image, f_point, l_point, (255,0,0), 2)

#이미지 보기
cv2.imshow('image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/draw_rectangle.jpg', image) # 파일 경로 & 명, 

#%% 이미지 잘라서 데이터셋의 형태로 만들어 보기 : 
# 이미지에 사각형을 그리는게 중요한게 아니라 좌표를 이용해서 이미지 값을 불러오는게 중요하다.

image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)


x1 = int(st_python['shapes'][0]['points'][0][0]) #0,0 첫좌표의 행값
y1 = int(st_python['shapes'][0]['points'][0][1]) #0,0 첫좌표의 열값
# 마지막 좌표
x2 = int(st_python['shapes'][0]['points'][1][0])
y2 = int(st_python['shapes'][0]['points'][1][1])
    

cropped_image = image[y1: y2, x1: x2].copy()
    
    
#이미지 보기
cv2.imshow('cropped_image',cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#이미지 저장하기
cv2.imwrite('../image/test_result/{}.jpg'.format(i), cropped_image) # 파일 경로 & 명, 

#%% 반복문의로 변경
image = cv2.imread('../image/SCAN_01.jpg', cv2.IMREAD_COLOR)
for i in range(len(st_python['shapes'])):
    # 첫 좌표
    x1 = int(st_python['shapes'][i]['points'][0][0]) #0,0 첫좌표의 행값
    y1 = int(st_python['shapes'][i]['points'][0][1]) #0,0 첫좌표의 열값
    # 마지막 좌표
    x2 = int(st_python['shapes'][i]['points'][1][0])
    y2 = int(st_python['shapes'][i]['points'][1][1])
    #크롭 이미지
    if x1 > x2:
        if y1 > y2:
            cropped_image = image[y2: y1, x2: x1].copy()    
        else:
            cropped_image = image[y1: y2, x2: x1].copy()    
    else:
        if y1 > y2:
            cropped_image = image[y2: y1, x1: x2].copy()    
        else:
            cropped_image = image[y1: y2, x1: x2].copy()    
    
    #이미지 저장하기
    cv2.imwrite('../image/test_result/{}.jpg'.format(i), cropped_image) # 파일 경로 & 명, 


