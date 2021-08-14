#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 14:39:18 2021

@author: kang
"""
import numpy as np

# a = np.array([[[1,2,3],[2,3,4],[3,4,5]], [[1,2,3],[2,3,4],[3,4,5]], [[1,2,3],[2,3,4],[3,4,5]]])
# b = np.array([[[0,0,2],[2,0,0],[1,1,1]], [[0,0,2],[2,0,0],[1,0,0]], [[0,0,2],[2,0,0],[0,0,1]]])
# print(a.shape)
# print(a)
# print(b)

# act_prod = np.einsum('kij,kij',a, b)
# print(act_prod)


a = [1,2,3,4,5,7,8]
b = [2,5,3,9,10]

print(a-b)