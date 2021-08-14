#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:31:32 2021

@author: kang
"""

import numpy as np
import json
import os
import matplotlib.pyplot as plt
from DatasetFFD import DatasetFFDNet
from model import FFDNet

import torch
import argparse
from utils_option import parse, make_logger



def vis(args):
    
    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['test']['logger_name']
    logger_path = args['test']['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    
    '''
    # ---------------------------
    # Dataset
    # ---------------------------
    # '''
    args_dataset = args['dataset']
    train_set = DatasetFFDNet(args_dataset['train'])
    test_set = DatasetFFDNet(args_dataset['test'])
        
    test_img1 = test_set[1]
    test_H1 = test_img1['H']    
    
    test_H1_view = np.uint8((test_H1.squeeze()*255).round())
    test_H1_view = np.transpose(test_H1_view, (1,2,0))
    test_H1_view = test_H1_view[:, :, [2,1,0]]
    
    
    test_img2 = test_set[5]
    test_H2 = test_img2['H']    
    
    test_H2_view = np.uint8((test_H2.squeeze()*255).round())
    test_H2_view = np.transpose(test_H2_view, (1,2,0))
    test_H2_view = test_H2_view[:, :, [2,1,0]]
    
    testset_name = args['dataset']['test']['test_set']
    
    
    '''
    # ---------------------------
    # Settings
    # ---------------------------
    '''
    epoch = args['method1']['epoch']
    batch_size = args['method1']['batch_size']
    mode = args['method1']['mode']
    is_clip = args['is_clip']
    
    logger.info(f'epoch:{epoch}, batch size:{batch_size}, is_clip:{is_clip}')
    
    
    base_path = 'mnist_if'
    file_root = 'if_'
    file_spec = 't1000r5'
    file_name = file_root + file_spec
    
    with open(os.path.join(base_path, file_name+'.json')) as fp:
        if_file = json.load(fp)
    
    if_file_n1 = if_file['0']
    if_value1 = if_file_n1['influence']
    top_images1 = if_file_n1['helpful'][:5]
    bad_images1 = if_file_n1['harmful'][:5]

    
    
    '''
    # ---------------------------
    # Plot images
    # ---------------------------
    '''
    plt.clf()
    for i, (top_num1, bad_num1) in enumerate(zip(top_images1, bad_images1)):
        
        top_train1 = train_set[top_num1]
        train_H1 = top_train1['H']
        train_H1_view = np.uint8((train_H1.squeeze()*255).round())
        train_H1_view = np.transpose(train_H1_view, (1,2,0))
        train_H1_view = train_H1_view[:, :, [2,1,0]]
        if_val1 = if_value1[top_num1]
        
        
        bad_train1 = train_set[bad_num1]
        train_B1 = bad_train1['H']
        train_B1_view = np.uint8((train_B1.squeeze()*255).round())
        train_B1_view = np.transpose(train_B1_view, (1,2,0))
        train_B1_view = train_B1_view[:, :, [2,1,0]]
        if_val2 = if_value1[bad_num1]
        
        
        
        plt.subplot(2,6,1)
        plt.title('test: baboon\nhelpful', fontdict={'fontsize':10}, y=-0.3)
        plt.axis('off')
        plt.imshow(test_H1_view)
        
        plt.subplot(2,6,i+2)
        plt.title('train_id: {}\nvalue: {:.2f}'.format(top_num1, if_val1), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_H1_view)
        
        plt.subplot(2,6,7)
        plt.title('test: baboon\nharmful', fontdict={'fontsize':10}, y=-0.3)
        plt.axis('off')
        plt.imshow(test_H1_view)
        
        plt.subplot(2,6,i+8)
        plt.title('train_id: {}\nvalue: {:.2f}'.format(bad_num1, if_val2), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_B1_view)
        
        
        
        
        
    if not os.path.exists(f'fig/m2/{mode}/{testset_name}'):
        os.makedirs(f'fig/m2/{mode}/{testset_name}')
    plt.savefig(f'fig/m2/{mode}/{testset_name}/baboon.png')

        
    
    
        

def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    
    vis(args)
    
    
    
    
if __name__ == '__main__':
    # vis_cifar()
    main()
    