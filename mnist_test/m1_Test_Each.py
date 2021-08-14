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

from utils_hg import parse
from utils_hg import make_logger
from model import mnist_model as net



def test(args):    
    
    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['logger_test']
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
    
    epochs = args['epochs']
    batch_size = args['batch_size'] 
    layer = args['layer']
    l_num = int((layer-1)*2+1)
    mode = args['mode']
    
    logger.info(f'Cuda: {args["cuda"]}')
    logger.info(f'mode: {mode}')
    logger.info(f'epoch:{epochs}, batch size:{batch_size}, layer:{layer}')
    
    
    '''
    # ----------------------
    # Datasets
    # ----------------------
    '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    test_img_num = args['test_img_num']
    test_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)

    logger.info('test_set: {}'.format(len(test_set)))
    
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    model_path = f'model_zoo/mnist_e{epochs}b{batch_size}_{mode}.pth'
    model = net()
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    if args['cuda']:
        model = model.cuda()
    
        
    
    '''
    # ----------------------
    # Calc Activations
    # ----------------------
    '''
    logger.info('------> Start testing...')
    
    test_set = test_set[test_img_num]
    image = test_set[0]
    label = test_set[1]
    
    image = image.view(image.shape[0], -1)
    
    if args['cuda']:
        image, label = image.cuda(), label.cuda()
        
    loaded_array = np.load(f'grad_norm/{mode}/e{epochs}b{batch_size}l{layer}.npy')    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    model.main[l_num].register_forward_hook(get_activation('main'))
    output = model(image)
    activation_val = activation['main'].numpy()
    matmul = np.matmul(activation_val, loaded_array.T)
    actgrad_list = matmul[0]
    
    
    
    '''
    # ----------------------
    # Save Activations
    # ----------------------
    '''
    if not os.path.exists(f'act_grad/{mode}'):
        os.mkdir(f'act_grad/{mode}')
    
    norm_path = f'act_grad/{mode}/t{test_img_num}l{layer}'
    np.save(norm_path, actgrad_list)
    logger.info(f'Saved act_grad values in {norm_path}')

    




def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    test(args)

    
    
if __name__ == '__main__':
    
    main()



