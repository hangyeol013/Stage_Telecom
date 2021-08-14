#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 09:43:03 2021

@author: kang
"""

import torch.nn as nn

#-------------------------------------
# Pixel Unshuffle
#-------------------------------------

def _Pixel_unshuffle(input_img, upscale_factor):
    
    [B, C, H, W] = list(input_img.shape)
    x = input_img.reshape(B, C, H//upscale_factor, upscale_factor, W//upscale_factor, upscale_factor)
    x = x.permute(0, 1, 3, 5, 2, 4)
    x = x.reshape(B, C*(upscale_factor**2), H//upscale_factor, W//upscale_factor)
    
    return x
    

class Pixel_unshuffle(nn.Module):
    
    def __init__(self, upscale_factor):
        super(Pixel_unshuffle, self).__init__()
        self.upscale_factor = upscale_factor
        
    def forward(self, input_img):
        return _Pixel_unshuffle(input_img=input_img, upscale_factor=self.upscale_factor)
        
        
#-------------------------------------
# Conv Block (Conv2d, BatchNorm, ReLU)
#-------------------------------------

def conv_block(channels, kernel_size, padding, stride, bias=True):
    
    
    conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, stride=stride,bias=bias)
    bnorm = nn.BatchNorm2d(num_features=channels)
    act_fn = nn.ReLU()
    
    return nn.Sequential(conv, bnorm, act_fn)


#-------------------------------------
# Describe model
#-------------------------------------

def describe_model(model):

    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) +'\n'
    msg += 'Net structure: {}'.format(str(model)) +'\n'
    return msg