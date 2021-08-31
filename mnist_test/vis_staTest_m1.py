#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""

import torch
import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt
import argparse
import numpy as np
import utils_hg as utils
from utils_hg import parse
from model import mnist_model as net

    


def visual(args):
    
    epoch = args['epoch']
    batch_size = args['batch_size']
    mode = args['method1']['mode']
    
    
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])

    train_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)

    test_imgs = [3520, 3879, 597, 2667, 5673, 8616, 5570, 9489, 3714, 7362]
    # staTest: you can choose from 0 to 5. (staTest0 ~ staTest5)
    staTest = [0,2,3,4]
    # num: you can choose from 0 to 9 (later, it will be used with test_imgs like test_imgs[num])
    num = 9
    
    
    path = f'model_zoo/mnist_e{epoch}b{batch_size}_staTest0.pth'
    model = net()
    if args['cuda']:
        model.load_state_dict(torch.load(path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    
    plt.clf()
    for i, idx in enumerate(staTest):    
    
        test_idx = test_imgs[num]
        test_name = test_idx
        test = test_set[test_idx]
        test_img = test[0]
        test_target = test[1]
        
        test_in = test_img.view(1,784)
        with torch.no_grad():
            logps = model(test_in)
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        
        args['base_path'] = f'act_grad/staTest{idx}/t{test_idx}lsum.npy'
        act_grad = np.load(args['base_path'])
        top_images = utils.topN_images_list(args)
        
        plt.subplot(len(staTest)+1,5,3)            
        plt.title(f'Test_id:{test_name}\ntarget:{test_target}\npredict:{pred_label}', fontdict={'fontsize':10}, y=-0.5)
        plt.axis('off')
        plt.imshow(test_img.numpy().squeeze(), cmap='gray_r')
        
        
        for j, top_name in enumerate(top_images):
            top_train = train_set[top_name]
            globals()[f'train_name{j}'] = top_name
            globals()[f'train_img{j}'] = top_train[0]
            globals()[f'train_target{j}'] = top_train[1]
            
            globals()[f'act_grad_v{j}'] = act_grad[top_name]
            
            
            plt.subplot(len(staTest)+1,5,5*i+6+j)
            if globals()[f'train_target{j}'] == pred_label:
                plt.title('train_id:{}\ntarget:{}\nvalue:{:.0f}'.format(globals()[f"train_name{j}"], globals()[f"train_target{j}"], globals()[f"act_grad_v{j}"]), fontdict = {'fontsize':9, 'color': 'red'}, y=-0.5)
            else:
                plt.title('train_id:{}\ntarget:{}\nvalue:{:.0f}'.format(globals()[f"train_name{j}"], globals()[f"train_target{j}"], globals()[f"act_grad_v{j}"]), fontdict = {'fontsize':9}, y=-0.5)
            plt.axis('off')
            plt.imshow(globals()[f'train_img{j}'].numpy().squeeze(), cmap='gray_r')
    
            
            





def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    args['vis'] = True

    visual(args)
        
    
    
if __name__ == '__main__':
    
    main()



