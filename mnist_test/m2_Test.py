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
    
    logger_name = args['logger_test']
    logger_path = args['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'dump: {args["method2"]["dump"]}')
    logger.info(f'stochastic: {args["method2"]["stochastic"]}')
    
    epoch = args['epoch']
    batch_size = args['batch_size']
    mode = args['method2']['mode']
    
    args['model_path'] = f'model_zoo/mnist_e{epoch}b{batch_size}_{mode}.pth'
    
    model = load_model(args)
    logger.info(f'model_zoo: {args["model_path"]}')
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



