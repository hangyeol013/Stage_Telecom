#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""


import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import argparse
import os
import random
import numpy as np

from utils_option import parse, make_logger
from utils_option import topLoss_img
from model import mnist_model as net



def test(args):    
    
    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['logger_train']
    logger_path = args['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    seed = args['seed']
    if seed is None:
        seed = random.randint(1, 100)
    logger.info('Random Seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
    '''
    # ----------------------
    # Datasets
    # ----------------------
    '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    test_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)
    # test_img_nums = topLoss_img(args)
    test_img_nums = [3520, 3879, 597, 2667, 5673, 8616, 5570, 9489, 3714, 7362]
    logger.info('Test_set: {}'.format(len(test_set)))
    logger.info(f'test_idx: {test_img_nums}')
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    model = net()
    criterion = nn.NLLLoss()
    
    nums = [200, 500, 1000, 2000, 5000, 10000]
    mode = args['newData']
    method = args['method']
    
    logger.info(f'Data set: {mode},    Method: {method}')
    logger.info(f'newData num: {nums}')
    
    
    '''
    # ----------------------
    # expTest Results
    # ----------------------
    '''
    logger.info('------> Start testing...')
    
    for num in nums:
        print(f'n_num: {num}')
        model_path_dir = f'model_zoo/expTest/seed{seed}/{method}_helpful/n{num}/{mode}'
        correct_count, all_count = 0,0
        wrong_list = []
    
        for file in os.listdir(model_path_dir):
            idx = file[1:-4]
            idx = int(idx)
            model_path = os.path.join(model_path_dir, file)
            if args['cuda']:
                model.load_state_dict(torch.load(model_path))
                model.cuda()
            else:
                model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            for name, param in model.named_parameters():
                param.requires_grad = False
                            
            test_img = test_set[idx]
            image = test_img[0]
            label = test_img[1]
            
            image = image.view(image.shape[0], -1)
            
            if args['cuda']:
                image, label = image.cuda(), label.cuda()
            
            logps = model(image)
            label_tor = torch.as_tensor([label])
            
            loss = criterion(logps, label_tor)
            
            ps = torch.exp(logps)
            probab = list(ps.cpu().numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = label
            
            print(f'loss_{idx}: ', loss,    f'true: {true_label}, pred: {pred_label}')
            
            if(true_label == pred_label):
              correct_count += 1
            else:
                # logger.info(f'wrong test_idx: {idx}')
                wrong_list.append(idx)
            all_count += 1
        
        print(f'wrong_list: {wrong_list}')
        print(f'Number of Images Testes = {all_count}')
        print(f'Model Accuracy(%) = {(correct_count/all_count) * 100}')
        # logger.info(f'Number of Images Testes = {all_count}')
        # logger.info(f'Model Accuracy(%) = {(correct_count/all_count) * 100}')   
    
    
    
    
    


def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    test(args)

    
    
if __name__ == '__main__':
    
    main()



