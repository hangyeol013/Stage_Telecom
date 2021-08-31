#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:18:27 2021

@author: hangyeol
"""

from torch.autograd import grad
import torch
import numpy as np
import random
import argparse
import json
from pathlib import Path

from utils_option import make_logger
from utils_option import parse, load_data, load_model
from utils_option import select_lossfn




def calc_loss(args, output, label):
    
    loss_fn = select_lossfn(opt=args['train']['loss_fn'], reduction=args['train']['reduction'])
    loss = loss_fn(output, label)
    
    return loss



def calc_grad(args, img_L, img_H, img_C, model):
    model.eval()
    if args['cuda']:
        img_L, img_H, img_C = img_L.cuda(), img_H.cuda(), img_C.cuda()
    
    y = model(img_L, img_C)
    loss = calc_loss(args, y, img_H)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    return list(grad(loss, params, create_graph=True))


'''
# ---------------------------------------
# Calc Hassian Vector Products
# ---------------------------------------
'''
def calc_hvp(outputs, weights, v):
    
    first_grads = grad(outputs=outputs, inputs=weights)
    
    elemwise_products = 0
    for grad_elem, v_elem in zip(first_grads, v):
        elemwise_products += torch.sum(grad_elem * v_elem)
    element_product = elemwise_products
    second_grad = grad(outputs=element_product, inputs=weights, retain_graph=True)
    return second_grad


'''
# ---------------------------------------
# Calc s_test
# ---------------------------------------
'''
def calc_s_test(args, test_L, test_H, test_C, model, train_loader, logger, recursion_nums=10, 
                training_points=5000, damp=0.01, scale=25.0):
    
    v = calc_grad(args=args, img_L=test_L, img_H=test_H, img_C=test_C, model=model)
    h_estimate = v.copy()
    
    
    s_test_list = []
    for i in range(recursion_nums):
    
        for j in range(training_points):
            
            for train_data in train_loader:
                
                train_L = train_data['L']
                train_H = train_data['H']
                train_C = train_data['C']
                
                if args['method2']['stochastic']:    
                    train_L = train_L[0]
                    train_H = torch.unsqueeze(train_H[0], 0)
                    
                if args['cuda']:
                    train_L, train_H, train_C = train_L.cuda(), train_H.cuda(), train_C.cuda()
                
                train_output = model(train_L, train_C)
                
                loss = calc_loss(args, train_output, train_H)
                params = [p for p in model.parameters() if p.requires_grad]
                hvp = calc_hvp(outputs=loss, weights=params, v=h_estimate)
                if args['method2']['damp']:
                    h_estimate = [_vector + (1 - damp) * _hessian_estimate - _hessian_vector / scale
                              for _vector, _hessian_estimate, _hessian_vector in zip(v, h_estimate, hvp)]
                else:
                    h_estimate = [_vector + _hessian_estimate - _hessian_vector
                                  for _vector, _hessian_estimate, _hessian_vector in zip(v, h_estimate, hvp)]
                
                break
            
            if j % 100 == 0:  
                logger.info(f"Calc. s_test training points: {j+1}/{training_points} ")
        logger.info(f'Calc, s_test reculsions: {i+1}/{recursion_nums}')
            
        s_test_list.append(h_estimate)
        
    s_test_vec = s_test_list[0]
    for k in range(1, recursion_nums):
        s_test_vec += s_test_list[k]
    s_test_vec = [v / recursion_nums for v in s_test_vec]
            
    return s_test_vec



def calc_influence_single(args, test_loader, test_id_num, model, 
                          train_loader, logger, s_test_vec=None):
    
    if not s_test_vec:
        test_L = test_loader.dataset[test_id_num]['L']
        test_H = test_loader.dataset[test_id_num]['H']
        test_C = test_loader.dataset[test_id_num]['C']
        
        test_L = test_loader.collate_fn([test_L])
        test_H = test_loader.collate_fn([test_H])
        test_C = test_loader.collate_fn([test_C])

        s_test_vec = calc_s_test(args, test_L=test_L, test_H=test_H, test_C=test_C, model=model, 
                                 train_loader=train_loader, logger=logger, 
                                 recursion_nums=args['method2']['recursion_nums'],
                                 training_points=args['method2']['training_points'])
    
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        train_L = train_loader.dataset[i]['L']
        train_H = train_loader.dataset[i]['H']
        train_C = train_loader.dataset[i]['C']

        train_L = train_loader.collate_fn([train_L])
        train_H = train_loader.collate_fn([train_H])
        train_C = train_loader.collate_fn([train_C])
        
        grad_v_vec = calc_grad(args=args, img_L=train_L, img_H=train_H, img_C = train_C, model=model)
        
        top_influence = -sum([torch.sum(k*j).data 
                              for k, j in zip(grad_v_vec, s_test_vec)])
        influences.append(top_influence)
        
        if i % 5000 == 0:        
            logger.info(f'Calc. influence functions: {i+1}/{train_dataset_size}')
    
    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    
    return influences, harmful.tolist(), helpful.tolist(), test_id_num
    
    
'''
# ---------------------------------------
# Calc influence functions
# ---------------------------------------
'''
def calc_influence_function(args, test_loader, train_loader, model, logger):
    
    outdir = Path(args['method2']['if_dir'])
    outdir.mkdir(exist_ok=True, parents=True)
    
    # test_sample_ids = topLoss_img(args)
    test_sample_num = args['method2']['test_sample_num']
    test_sample_ids = random.sample(range(len(test_loader.dataset)), test_sample_num)
    test_sample_ids = [1, 5]
    logger.info(f'Test sample num: {test_sample_num}')
    logger.info(f'{test_sample_ids}')
    
    logger.info(f"Recursion_nums: {args['method2']['recursion_nums']}")
    logger.info(f"Training_points: {args['method2']['training_points']}")
    logger.info(f"batch_size: {args['method2']['batch_size']}")

    influences = {}
    logger.info('-------> Start')
    for i, test_id in enumerate(test_sample_ids):
        logger.info(f'test_id: {test_id} ({i+1}/{test_sample_num})')
        influence, harmful, helpful, \
        test_name = calc_influence_single(args=args, test_loader=test_loader, 
                                          test_id_num=test_id, model=model, 
                                          train_loader=train_loader, logger=logger)
        
        infl = [x.cpu().numpy().tolist() for x in influence]
        
        influences[str(i)] = {}
        influences[str(i)]['test_id_num'] = test_id
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:10000]
        influences[str(i)]['helpful'] = helpful[:10000]
        
        logger.info(f'Test images: {i+1}/{len(test_sample_ids)}')
        
        logger.info(f'Influences: {influences[str(i)]["influence"][:3]}')
        logger.info(f'Most harmful img IDs: {influences[str(i)]["harmful"][:5]}')
        logger.info(f'Most helpful img IDs: {influences[str(i)]["helpful"][:5]}')
    
    
    if args['method2']['stochastic']:
        influences_path = outdir.joinpath(f'if_t{args["method2"]["training_points"]}r{args["method2"]["recursion_nums"]}b{1}.json')
    else:        
        influences_path = outdir.joinpath(f'if_t{args["method2"]["training_points"]}r{args["method2"]["recursion_nums"]}b{args["method2"]["batch_size"]}_new.json')
    
    
    
    with open(influences_path, 'w') as outfile:
        json.dump(influences, outfile)
        
    
    return influences




def main(json_path = 'Implementation.json'):
    
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    logger_name = args['test']['logger_name']
    logger_path = args['test']['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    model = load_model(args)
    trainloader, testloader = load_data(args)
    
    calc_influence_function(args=args, test_loader=testloader, train_loader=trainloader, 
                            model=model, logger=logger)
    


if __name__ == '__main__':
    
    main()
    
    
