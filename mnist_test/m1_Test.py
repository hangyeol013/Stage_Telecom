#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""


import torch
import torchvision
from torchvision import transforms
import argparse
import numpy as np
import random
import os

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

    epoch = args['epoch']
    batch_size = args['batch_size']
    mode = args['method1']['mode']
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'mode: {mode}')
    logger.info(f'epoch:{epoch}, batch size:{batch_size}, layer:All')
    
    
    
    '''
    # ----------------------
    # Datasets
    # ----------------------
    '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    test_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)

    logger.info('test_set: {}'.format(len(test_set)))
    
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    model_path = f'model_zoo/mnist_e{epoch}b{batch_size}_{mode}.pth'
    
    model = net()
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    if args['cuda']:
        model.load_state_dict(torch.load(model_path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    # test_img_idxs = topLoss_img(args)
    # test_img_idxs = [3520, 3879, 597, 2667, 5673, 8616, 5570, 9489, 3714, 7362]
    test_img_idxs = [1031, 4406]
    
    
    '''
    # ----------------------
    # Calc Activations
    # ----------------------
    '''
    logger.info('------> Start testing...')
    for test_num in test_img_idxs:
        test_img = test_set[test_num]
    
        image = test_img[0]
        label = test_img[1]
        
        image = image.view(image.shape[0], -1)
        
        if args['cuda']:
            image, label = image.cuda(), label.cuda()
            
        loaded_array1 = np.load(f'grad_norm/{mode}/e{epoch}b{batch_size}l1.npy')   
        loaded_array2 = np.load(f'grad_norm/{mode}/e{epoch}b{batch_size}l2.npy')
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        l1 = 1
        l2 = 3
        model.main[l1].register_forward_hook(get_activation('main1'))
        model.main[l2].register_forward_hook(get_activation('main2'))
        
        output = model(image)
        activation_val1 = activation['main1'].cpu().numpy()
        activation_val2 = activation['main2'].cpu().numpy()
        
        activation_val1[activation_val1 < 0] = 0
        activation_val2[activation_val2 < 0] = 0
        
        matmul1 = np.matmul(activation_val1, loaded_array1.T)
        matmul2 = np.matmul(activation_val2, loaded_array2.T)
        
        actgrad_list1 = matmul1[0]
        actgrad_list2 = matmul2[0]
        
        actgrad_sum = actgrad_list1 + actgrad_list2
        
        
        
        '''
        # ----------------------
        # Save Activations
        # ----------------------
        '''
        if not os.path.exists(f'act_grad/{mode}'):
            os.makedirs(f'act_grad/{mode}')
        
        norm_path = f'act_grad/{mode}/t{test_num}lsum'
        np.save(norm_path, actgrad_sum)
        logger.info(f'Saved act_grad values in {norm_path}')

    


def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    test(args)

    
    
if __name__ == '__main__':
    
    main()



