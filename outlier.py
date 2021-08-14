#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 10:56:39 2021

@author: kang
"""


import numpy as np
import matplotlib.pyplot as plt

    
    
loaded_array = np.load('grad_norm/e20b8l11_1000.npy')
loaded_sum = np.sum(loaded_array, axis=3)
loaded_sum = np.sum(loaded_sum, axis=2)
print(loaded_sum.shape)
gn_sum = np.sum(loaded_sum, axis=1)
gn_16 = loaded_sum[:, 20]

print(gn_16.shape)

plt.clf()
# plt.plot(gn_sum)
plt.hist(gn_sum)
plt.xlabel('gradient norm', size=15)
plt.ylabel('number of images', size=15)
plt.show()
# plt.xticks(np.arange(0, 20, 0.5))


x = [k for k in range(len(gn_sum)) if gn_sum[k] > 13]
print('big_val idx: ', x)
print('big_val num: ', len(x))

imgs = [25917, 38963, 10294, 6953, 20860, 5173, 36434, 45293, 10806, 19406, 
        19622, 6778, 45510, 39891, 34947]
imgs1 = [25917, 38963, 10294, 6953, 20860]
imgs_set = set(imgs)
imgs1_set = set(imgs1)
int01 = imgs_set.intersection(x)
int02 = imgs1_set.intersection(x)
print('int_imgs: ', int01)
print('int_imgs1: ', int02)
print('imgs_len: ', len(imgs))
print('int_len: ', len(int01))


