#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:18:27 2021

@author: hangyeol
"""

from torch.autograd import grad
from torch import nn
import torch
import numpy as np
import random
import argparse
import json
from pathlib import Path

from utils_option import parse, load_data, load_model
from utils_option import get_ids_per_class, topLoss_img




def calc_loss(y, label):
    criterion = nn.NLLLoss()
    loss = criterion(y, label)
    
    return loss



def calc_grad(args, img, label, model):
    model.eval()
    img = img.view(img.shape[0], -1)
    if args['cuda']:
        img, label = img.cuda(), label.cuda()
    
    y = model(img)
    loss = calc_loss(y, label)
    
    params = [p for p in model.parameters() if p.requires_grad]
    
    return list(grad(loss, params, create_graph=True))


'''
# ---------------------------------------
# Calc Hassian Vector Products (WHY ?!!!!!!!!! SECOND GRAD ELEMENTWISE PRODUCTS)
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
# Calc s_test (WHY ?!!!!!!!!! What is damp and scale ?!!!!)
# ---------------------------------------
'''
def calc_s_test(args, test_img, test_label, model, train_loader, logger, recursion_nums=10, 
                training_points=5000, damp=0.01, scale=25.0):
    
    v = calc_grad(args=args, img=test_img, label=test_label, model=model)
    h_estimate = v.copy()
    
    
    s_test_list = []
    for i in range(recursion_nums):
    
        for j in range(training_points):
            
            for train_img, train_label in train_loader:
                
                if args["method2"]['stochastic']:    
                    train_img = train_img[0]
                    train_label = torch.unsqueeze(train_label[0], 0)
                    
                if args['cuda']:
                    train_img, train_label, model = train_img.cuda(), train_label.cuda(), model.cuda()
                
                train_img = train_img.view(train_img.shape[0], -1)
                train_output = model(train_img)
                
                loss = calc_loss(train_output, train_label)
                params = [p for p in model.parameters() if p.requires_grad]
                hvp = calc_hvp(outputs=loss, weights=params, v=h_estimate)
                if args["method2"]['dump']:
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
        test_img, test_label = test_loader.dataset[test_id_num]
        test_img = test_loader.collate_fn([test_img])
        test_label = test_loader.collate_fn([test_label])
        s_test_vec = calc_s_test(args, test_img=test_img, test_label=test_label, model=model, 
                                 train_loader=train_loader, logger=logger, 
                                 recursion_nums=args["method2"]['recursion_nums'],
                                 training_points=args["method2"]['training_points'])
    
    train_dataset_size = len(train_loader.dataset)
    influences = []
    for i in range(train_dataset_size):
        train_img, train_label = train_loader.dataset[i]
        train_img = train_loader.collate_fn([train_img])
        train_label = train_loader.collate_fn([train_label])
        
        grad_v_vec = calc_grad(args=args, img=train_img, label=train_label, model=model)
        
        tmp_influence = -sum([torch.sum(k*j).data 
                              for k, j in zip(grad_v_vec, s_test_vec)])
        influences.append(tmp_influence)
        
        if i % 5000 == 0:        
            logger.info(f'Calc. influence functions: {i+1}/{train_dataset_size}')
    
    harmful = np.argsort(influences)
    helpful = harmful[::-1]
    
    return influences, harmful.tolist(), helpful.tolist(), test_id_num, test_label.item()
    
    
'''
# ---------------------------------------
# Calc influence functions
# ---------------------------------------
'''
def calc_influence_function(args, test_loader, train_loader, model, logger):
    
    outdir = Path(args['method2']['if_dir'])
    outdir.mkdir(exist_ok=True, parents=True)
    
    # test_sample_ids = topLoss_img(args)
    test_sample_ids = [4406]
    test_sample_num = len(test_sample_ids)
    logger.info(f'Test sample num: {test_sample_num}')
    logger.info(f'{test_sample_ids}')
    
    logger.info(f"Recursion_nums: {args['method2']['recursion_nums']}")
    logger.info(f"Training_points: {args['method2']['training_points']}")
    logger.info(f"batch_size: {args['batch_size']}")

    influences = {}
    logger.info('-------> Start')
    for i, test_id in enumerate(test_sample_ids):
        logger.info(f'test_id: {test_id} ({i+1}/{test_sample_num})')
        influence, harmful, helpful, \
        test_name, test_label = calc_influence_single(args=args, test_loader=test_loader, 
                                                      test_id_num=test_id, model=model, 
                                                      train_loader=train_loader, logger=logger)
        
        infl = [x.cpu().numpy().tolist() for x in influence]
        
        influences[str(i)] = {}
        influences[str(i)]['test_id_num'] = test_id
        influences[str(i)]['label'] = test_label
        influences[str(i)]['influence'] = infl
        influences[str(i)]['harmful'] = harmful[:10000]
        influences[str(i)]['helpful'] = helpful[:10000]
        
        logger.info(f'Test images: {i+1}/{len(test_sample_ids)}')
        
        logger.info(f'Influences: {influences[str(i)]["influence"][:3]}')
        logger.info(f'Most harmful img IDs: {influences[str(i)]["harmful"][:5]}')
        logger.info(f'Most helpful img IDs: {influences[str(i)]["helpful"][:5]}')
    
    
    if args['method2']['stochastic']:
        influences_path = outdir.joinpath(f'if_t{args["method2"]["training_points"]}r{args["method2"]["recursion_nums"]}b{1}_{args["method2"]["mode"]}.json')
    else:        
        influences_path = outdir.joinpath(f'if_t{args["method2"]["training_points"]}r{args["method2"]["recursion_nums"]}b{args["batch_size"]}_{args["method2"]["mode"]}_4406.json')
    
    
    with open(influences_path, 'w') as outfile:
        json.dump(influences, outfile)
    
    
    
    return influences




def main(json_path = 'Implementation.json'):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    model = load_model(args)
    trainloader, testloader = load_data(args)
    
    calc_influence_function(args=args, test_loader=testloader, train_loader=trainloader, model=model)
    


if __name__ == '__main__':
    
    main()
    
    
