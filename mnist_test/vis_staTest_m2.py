#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 09:31:32 2021

@author: kang
"""

import json
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision
import torch
import argparse
from utils_option import parse
from utils_option import load_model



def vis(args):
    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    
    train_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=True, transform=transform)   
    test_set = torchvision.datasets.MNIST(root='./datasets', download=True, train=False, transform=transform)

    test_imgs = [2, 7, 5, 6, 8, 9, 3, 1]
    num = 7
    
    # test_imgs = [0,1]
    # num = 0
    
    staTest = [0,3,4,5]
    # staTest = [0]
    
    batch_size = args['batch_size']
    recursion_nums = args['method2']['recursion_nums']
    training_points = args['method2']['training_points']
    
    model = load_model(args)
    base_path = 'mnist_if'
    file_root = 'if_'
    
    plt.clf()
    for i, idx in enumerate(staTest):    
        file_spec = f't{training_points}r{recursion_nums}b{batch_size}_staTest{idx}'
        file_name = file_root + file_spec
        
        with open(os.path.join(base_path, file_name+'.json')) as fp:
            if_file = json.load(fp)
            
        if_file_n = if_file[str(test_imgs[num])]
        test_idx = if_file_n['test_id_num']
        test = test_set[test_idx]
        test_name = test_idx
        test_img = test[0]
        test_target = test[1]
    
        test_in = test_img.view(1,784)
        with torch.no_grad():
            logps = model(test_in)
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        
        top_images = if_file_n['helpful'][:5]
        if_value = if_file_n['influence']
        print(file_spec, top_images)
        
        plt.subplot(len(staTest)+1,5,3)            
        plt.title(f'Test_id:{test_name}\ntarget:{test_target}\npredict:{pred_label}', fontdict={'fontsize':10}, y=-0.5)
        plt.axis('off')
        plt.imshow(test_img.numpy().squeeze(), cmap='gray_r')
        
        
        for j, top_name in enumerate(top_images):
            top_train = train_set[top_name]
            globals()[f'train_name{j}'] = top_name
            globals()[f'train_img{j}'] = top_train[0]
            globals()[f'train_target{j}'] = top_train[1]
            
            globals()[f'act_grad_v{j}'] = if_value[top_name]
            
            
            plt.subplot(len(staTest)+1,5,5*i+6+j)
            if globals()[f'train_target{j}'] == pred_label:
                plt.title('train_id:{}\ntarget:{}\nvalue:{:.2f}'.format(globals()[f"train_name{j}"], globals()[f"train_target{j}"], globals()[f"act_grad_v{j}"]), fontdict = {'fontsize':9, 'color': 'red'}, y=-0.5)
            else:
                plt.title('train_id:{}\ntarget:{}\nvalue:{:.2f}'.format(globals()[f"train_name{j}"], globals()[f"train_target{j}"], globals()[f"act_grad_v{j}"]), fontdict = {'fontsize':9}, y=-0.5)
            plt.axis('off')
            plt.imshow(globals()[f'train_img{j}'].numpy().squeeze(), cmap='gray_r')
    
    
    
    
        
    
    
        

def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    
    vis(args)
    
    
    
    
if __name__ == '__main__':
    # vis_cifar()
    main()
    