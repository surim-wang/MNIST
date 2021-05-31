# -*- coding: utf-8 -*-
"""
Created on Mon May 31 22:19:20 2021

@author: SURIMWANG
"""

#%%
import cv2
import numpy as np
import os
os.chdir('D:/MNIST/source')

image = cv2.imread('../image/smiley.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
result = np.zeros((image.shape[0], 256), dtype=np.uint8) 
# np.zeros()함수를 이용해 x행,y열의 구조를 만드는데 값은 0이 들어간다.
# dtype=np.uint8으로 정했는데 gray 스케일에서 가장 잘 사용하는 bit depth이다(정밀도) 2^8을 말하는것으로 하나의 비트를 256개의 색으로 입력할수있다.
test = np.zeros((2,2))
