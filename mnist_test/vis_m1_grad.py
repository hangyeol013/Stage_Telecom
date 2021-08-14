#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""

import torch

import matplotlib.pyplot as plt
import argparse
import numpy as np

import utils_option as utils
from utils_option import parse

import torchvision
from torchvision import transforms


def visual(args):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    train_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=True, transform=transform)
    
    args['method'] = 'method0'
    epoch = args['epoch']
    batch_size = args['batch_size']
    
    base_path = f'grad_norm/staTest0/e{epoch}b{batch_size}l3.npy'
    args['base_path'] = base_path
    
    loaded_grad = np.load(base_path)
    print(loaded_grad.shape)
    
    count = 0
    plt.clf()
    for node in range(5,10):
        args['node'] = node
        top_images = utils.topN_images_list(args)
        print(top_images)
        
        for i, top_name in enumerate(top_images):
            grad_v = loaded_grad[:, node]
            top_image = train_set[top_name]
            img = top_image[0]
            target = top_image[1]
            
            grad = grad_v[top_name]
            
            plt.suptitle(f'Top 5 of last layer (e:{epoch}, b:{batch_size})')
            plt.subplot(5,5, count+1)
            if target == node:
                plt.title('train_id:{}\ntarget:{}\ngrad:{:.2f}'.format(top_name, target, grad), fontdict = {'fontsize':10, 'color': 'red'}, y=-0.5)
            else:
                plt.title('train_id:{}\ntarget:{}\ngrad:{:.2f}'.format(top_name, target, grad), fontdict = {'fontsize':10}, y=-0.5)
            plt.axis('off')
            plt.imshow(img.numpy().squeeze(), cmap='gray_r')
            count += 1
            


def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    args['vis'] = True

    visual(args)
        
    
    
if __name__ == '__main__':
    
    main()



