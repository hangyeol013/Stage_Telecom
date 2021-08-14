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

    # test_imgs = [597, 9489, 1031, 4406, 8616, 5570, 3714, 7362, 2667, 3879]
    # test_imgs = [597, 9489, 1031, 4406, 8616]
    test_imgs = [5570, 3714, 7362, 2667, 3879]


    path = f'model_zoo/mnist_e{epoch}b{batch_size}_staTest0.pth'
    model = net()
    if args['cuda']:
        model.load_state_dict(torch.load(path))
        model.cuda()
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    
    plt.clf()
    for i, test_idx in enumerate(test_imgs):
        test = test_set[test_idx]
        globals()[f'test_name{i}'] = test_idx
        globals()[f'test_img{i}'] = test[0]
        globals()[f'test_target{i}'] = test[1]
        
        test_in = globals()[f'test_img{i}'].view(1,784)
        with torch.no_grad():
            logps = model(test_in)
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        globals()[f'pred_label{i}'] = probab.index(max(probab))
        
        args['base_path'] = f'act_grad/staTest0/t{test_idx}lsum.npy'
        act_grad = np.load(args['base_path'])
        top_images = utils.topN_images_list(args)
        
        plt.subplot(5,6,6*i+1)            
        plt.title(f'Test_id:{globals()[f"test_name{i}"]}\ntarget:{globals()[f"test_target{i}"]}\npredict:{globals()[f"pred_label{i}"]}', fontdict={'fontsize':10}, y=-0.5)
        plt.axis('off')
        plt.imshow(globals()[f'test_img{i}'].numpy().squeeze(), cmap='gray_r')
        
        
        for j, top_name in enumerate(top_images):
            top_train = train_set[top_name]
            globals()[f'train_name{j}'] = top_name
            globals()[f'train_img{j}'] = top_train[0]
            globals()[f'train_target{j}'] = top_train[1]
            
            globals()[f'act_grad_v{j}'] = act_grad[top_name]
            
            
            plt.subplot(5,6,6*i+2+j)
            if globals()[f'train_target{j}'] == globals()[f'pred_label{i}']:
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



