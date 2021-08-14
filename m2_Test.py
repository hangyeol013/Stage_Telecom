#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""


import torch
import argparse

from utils_if import calc_influence_function
from utils_option import parse, make_logger
from utils_option import load_data, load_model



'''
# --------------------------
# Calc Influence Functions
# --------------------------
'''
def test(args):
    
    logger_name = args['test']['logger_name']
    logger_path = args['test']['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    
    epoch = args['method2']['epoch']
    batch_size = args['method2']['batch_size']
    
    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}.pth'
    else:
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}_1000.pth'
    
    model = load_model(args)
    logger.info(f'model: {model_path}')
    trainloader, testloader = load_data(args)
    
    calc_influence_function(args=args, test_loader=testloader, train_loader=trainloader, model=model,
                            logger=logger)
    
    




def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()


    test(args)

    
    
if __name__ == '__main__':
    
    main()



