#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 03 11:10:30 2021

@author: kang
"""

import os
import json
import random
import logging
import numpy as np
import argparse

from datasets.dataset_remove import MNIST as mnist_remove
from model import mnist_model as Net

from torchvision import transforms
import torchvision
import torch.nn as nn
from model import mnist_model as net
import torch



'''
# ----------------------------
# Parse
# ----------------------------
'''
def parse(opt_path):
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str)
    return opt

'''
# ----------------------------
# Make logger
# ----------------------------
'''
def make_logger(file_path, name=None):
    logger = logging.getLogger(name)
    
    if not logger.hasHandlers():
    
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(fmt='%(asctime)s - %(message)s')
        
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        
        # file_handler = logging.FileHandler(filename=file_path, mode='w')
        # file_handler.setLevel(logging.DEBUG)
        # file_handler.setFormatter(formatter)
        # logger.addHandler(file_handler) 

    return logger



'''
# -------------------------
# Load Model
# -------------------------
'''
def load_model(args):
    net = Net()
    path = args['model_base_path']
    if args['cuda']:
        net.load_state_dict(torch.load(path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return net



'''
# -------------------------
# Load Data (for IF)
# -------------------------
'''
def load_data(args):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    train_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, 
                                              num_workers=4, shuffle=True)
    
    test_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, 
                                             num_workers=4, shuffle=False)
    
    return trainloader, testloader



'''
# -------------------------
# Get idx per class
# -------------------------
'''
def get_ids_per_class(test_loader, start_idx, num_samples):
    
    num_list = []
    class_list = []
    img_count = 0
    for i in range(start_idx, len(test_loader.dataset)):
        _, c = test_loader.dataset[i]
        if c not in class_list:
            img_count += 1
            class_list.append(c)
            if img_count <= num_samples:
                num_list.append(i)
            else:
                break
    return num_list



'''
# -------------------------
# Top Loss img list
# -------------------------
'''
def topLoss_img(args):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                   transforms.Normalize((0.5,), (0.5,))])
    testset = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    path = args['model_base_path']
    model = net()
    if args['cuda']:
        model.load_state_dict(torch.load(path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))        
    criterion = nn.NLLLoss()
    
    losses = []
    
    for test_n in testloader:
        
        n_img = test_n[0]
        n_label = test_n[1]
        
        if args['cuda']:
            n_img = n_img.cuda()
            n_label = n_label.cuda()
        
        n_img = n_img.view(n_img.shape[0], -1)
        output = model(n_img)
        loss = criterion(output, n_label)
        losses.append(loss.item())
        
    top_loss = sorted(range(len(losses)), key=lambda k:losses[k], reverse=True)[0:10000:1000]
    
    # print(top_loss)

    return top_loss
    
    
    
'''
# -------------------------
# TopN images list
# -------------------------
'''
def topN_images_list(args):
    
    base_path = args['base_path']
    loaded_v = np.load(base_path)
    
    if args['method'] == 'method0':
        v = loaded_v[:, args['node']]
        top_images = sorted(range(len(v)), key = lambda k: v[k], reverse=True)[:5]
    else:
        v = loaded_v
        top_images = sorted(range(len(v)), key = lambda k: v[k], reverse=True)[:args['vis_num']]
    
    return top_images
    
    
    
'''
# -------------------------
# New Datasets For ExpTest
# -------------------------
''' 
def newDataset(args, test_idx, i):
    
    transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    test_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=False, transform=transform)
    
    
    if args['method'] == 'method1':    
        staTest = 'staTest0'
        args['base_path'] = f'act_grad/{staTest}/t{test_idx}lsum.npy'
        test_label = test_set[test_idx][1]
        top_images = topN_images_list(args=args)
    elif args['method'] == 'method2':
        base_path = 'mnist_if'
        file_name = 'if_t1000r5b8_staTest0'
        
        with open(os.path.join(base_path, file_name+'.json')) as fp:
            if_file = json.load(fp)
            
        if_file_n = if_file[i]
        test_label = if_file_n['label']
        top_images = if_file_n['helpful'][:args['vis_num']]
    
    
    if args['newData'] == 'algo':    
        train_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
        for img in top_images:
            idx = img
            train_set.targets[idx] = 0
            if train_set.targets[idx] == test_label:
                train_set.targets[idx] = 1
    
    
    if args['newData'] == 'algo2':
        train_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, 
                                               transform=transform)
        for img in top_images:
            idx = img
            if test_label == 0:
                if train_set.targets[idx] == test_label:
                    train_set.targets[idx] = 1
                else:
                    if train_set.targets[idx] == 1:
                        train_set.targets[idx] = 0
            else:
                if train_set.targets[idx] == test_label:
                    train_set.targets[idx] = 0
                else:
                    if train_set.targets[idx] == 0:
                        train_set.targets[idx] = 1
            
            
    elif args['newData'] == 'random':
        train_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)
        idx_list = []
        for _ in range(len(top_images)*2):
            r_idx = random.randint(0, 59999)
            idx_list.append(r_idx)
            if len(set(idx_list)) == len(top_images):
                break
        for idx in idx_list:
            train_set.targets[idx] = 0
            if train_set.targets[idx] == test_set[test_idx][1]:
                train_set.targets[idx] = 1
    
    
    elif args['newData'] == 'remove_algo':
        train_set = mnist_remove(remove_list=top_images, root = './datasets/algo', download=True, train=True, 
                                 transform=transform)
        
        
    elif args['newData'] == 'remove_random':
        rand_list = []
        for _ in range(len(top_images)*2):
            idx = random.randint(0, 59999)
            rand_list.append(idx)
            if len(set(rand_list)) == len(top_images):                
                break
        train_set = mnist_remove(remove_list=rand_list, root = './datasets/ran', download=True, train=True, 
                                 transform=transform)
    
        
    return train_set







def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    topLoss_img(args)
    # args['vis'] = True
    # args['base_path'] = 'act_grad/staTest1/t6196e20b8lsum.npy'
    # topN_images(args)
    # staTest = 'staTest0'
    # args['base_path'] = f'act_grad/{staTest}/t141lsum.npy'
    # topN_images_list(args)
    
    
if __name__ == '__main__':
    
    main()



