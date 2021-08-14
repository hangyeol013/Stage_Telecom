#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""

import torch.nn as nn
import torch


class mnist_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_size = 784
        self.hidden_sizes = [128,64]
        self.output_size = 10
        
        L = []
        L.append(nn.Linear(in_features=self.input_size, out_features=self.hidden_sizes[0]))
        L.append(nn.ReLU())
        L.append(nn.Linear(in_features=self.hidden_sizes[0], out_features=self.hidden_sizes[1]))
        L.append(nn.ReLU())
        L.append(nn.Linear(in_features=self.hidden_sizes[1], out_features=self.output_size))
        
        self.main = nn.Sequential(*L)
        
        self.softmax = nn.LogSoftmax(dim=1)

        
    def forward(self, img):
        
        x = self.main(img)
        x = self.softmax(x)
        
        return x
    
    
if __name__ == '__main__':
    
    model = mnist_model()

    print(model)
    
    x = torch.ones([1,784])
    y = model(x)
    print(y.shape)
    
    # print(model.main[2:4])
    
    # random.seed(1)
    # np.random.seed(1)
    # torch.manual_seed(1)
    
    # activation = {}
    # def get_activation(name):
    #     def hook(model, input, output):
    #         activation[name] = output.detach()
    #     return hook
    
    
    # model.main[2].register_forward_hook(get_activation('m'))
    # output = model(x)
    # print(activation['m'])
    
    # model.main[3].register_forward_hook(get_activation('main'))
    # output = model(x)
    # print(activation['main'])
    
    
    # for name, parameter in model.named_parameters():
    #     print(name, parameter.shape)
    #     if name == 'main.2.weight':
    #         print('HG: ', parameter.shape)
        




