#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 09:21:41 2021

@author: kang
"""

import numpy as np
import argparse
import torch
import json
import os

from utils_option import parse
import utils_option as utils




def staTest(args):
    
    test_imgs = [3520, 3879, 597, 2667, 5673, 8616, 5570, 9489, 3714, 7362]
    
    print('----- method1 ------')
    for test_idx in test_imgs:
        
        for i in range(0,6):
            args['base_path'] = f'act_grad/staTest{i}/t{test_idx}lsum.npy'
            globals()[f'top{i}_images'] = utils.topN_images_list(args=args)
            globals()[f'top{i}_images'] = globals()[f'top{i}_images'][:100]
                
        print('test_idx: ', test_idx)
        sta0_set = set(top0_images)
        int01 = sta0_set.intersection(top1_images)
        int02 = sta0_set.intersection(top2_images)
        int03 = sta0_set.intersection(top3_images)
        int04 = sta0_set.intersection(top4_images)
        int05 = sta0_set.intersection(top5_images)
        
        print('01: ', len(int01))
        print('02: ', len(int02))
        print('03: ', len(int03))
        print('04: ', len(int04))
        print('05: ', len(int05))
        print('-----------')



    print('----- method2 (helpful) ------')
    base_path = 'mnist_if'
    file_root = 'if_'
    file_spec = 't1000r5b8_staTest'
    file_name = file_root + file_spec
    
    for i in range(0,6):
        file_n = file_name + f'{i}'
        
        with open(os.path.join(base_path, file_n+'.json')) as fp:
            if_file = json.load(fp)
            
        for nums in range(0,10):
            if_file_n = if_file[str(nums)]
            test_id_num = if_file_n['test_id_num']
            globals()[f'sta{i}_img{nums}'] = if_file_n['helpful'][:100]
        
    for j in range(0,10):
        
        if_file_n = if_file[str(j)]
        test_id_num = if_file_n['test_id_num']
        
        print('test_idx: ', test_id_num)
        
        sta0_set = set(globals()[f'sta0_img{j}'])
        int01 = sta0_set.intersection(globals()[f'sta1_img{j}'])
        int02 = sta0_set.intersection(globals()[f'sta2_img{j}'])
        int03 = sta0_set.intersection(globals()[f'sta3_img{j}'])
        int04 = sta0_set.intersection(globals()[f'sta4_img{j}'])
        int05 = sta0_set.intersection(globals()[f'sta5_img{j}'])
        
        print('01: ', len(int01))
        print('02: ', len(int02))
        print('03: ', len(int03))
        print('04: ', len(int04))
        print('05: ', len(int05))
        print('----------')
        
    
    print('----- method2 (harmful) ------')
    base_path = 'mnist_if'
    file_root = 'if_'
    file_spec = 't1000r5b8_staTest'
    file_name = file_root + file_spec
    
    for i in range(0,6):
        file_n = file_name + f'{i}'
        
        with open(os.path.join(base_path, file_n+'.json')) as fp:
            if_file = json.load(fp)
            
        for nums in range(0,10):
            if_file_n = if_file[str(nums)]
            test_id_num = if_file_n['test_id_num']
            globals()[f'sta{i}_img{nums}'] = if_file_n['harmful'][:100]

        
    for j in range(0,10):
        
        if_file_n = if_file[str(j)]
        test_id_num = if_file_n['test_id_num']
        
        print('test_idx: ', test_id_num)
        
        sta0_set = set(globals()[f'sta0_img{j}'])
        int01 = sta0_set.intersection(globals()[f'sta1_img{j}'])
        int02 = sta0_set.intersection(globals()[f'sta2_img{j}'])
        int03 = sta0_set.intersection(globals()[f'sta3_img{j}'])
        int04 = sta0_set.intersection(globals()[f'sta4_img{j}'])
        int05 = sta0_set.intersection(globals()[f'sta5_img{j}'])
        
        print('01: ', len(int01))
        print('02: ', len(int02))
        print('03: ', len(int03))
        print('04: ', len(int04))
        print('05: ', len(int05))
        print('----------')

    



def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    args['vis'] = True

    staTest(args)
        
    
    
if __name__ == '__main__':
    
    main()
