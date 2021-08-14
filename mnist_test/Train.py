#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""


import torch
from torch import nn, optim
from torchvision import transforms
import argparse
from numpy.linalg import norm
import numpy as np
import os
import random

from utils_option import parse, make_logger
from model import mnist_model as net
from datasets.datasets_name import MNIST as mnist_name




def train(args):
    
    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['logger_train']
    logger_path = args['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    # seed = args['seed']
    # if seed is None:
    #     seed = random.randint(1, 100)
    # logger.info('Random Seed: {}'.format(seed))
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    
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
    train_set = mnist_name(root = './datasets', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    
    test_set = mnist_name(root = './datasets', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    
    logger.info('train_set: {}'.format(len(train_set)))
    logger.info('test_set: {}'.format(len(test_set)))
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    model = net()
    
    if args['cuda']:
        model = model.cuda()
    
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=0.001)
    
    for name, parameter in model.named_parameters():
        # if name == 'main.0.weight':
        #     grad_shape1 = parameter.shape[0]
        #     grad_norm_list1 = np.zeros([len(train_set), grad_shape1])
        # if name == 'main.2.weight':
        #     grad_shape2 = parameter.shape[0]
        #     grad_norm_list2 = np.zeros([len(train_set), grad_shape2])
        if name == 'main.4.weight':
            grad_shape3 = parameter.shape[0]
            grad_norm_list3 = np.zeros([len(train_set), grad_shape3])


    '''
    # ----------------------
    # Training & Calc Grad
    # ----------------------
    '''
    logger.info('------> Start Training...')
    for e in range(epoch):
        training_loss = 0
        for i, mnist_data in enumerate(trainloader):
            
            img_names = mnist_data['name']
            images = mnist_data['img']
            labels = mnist_data['target']
            
            images = images.view(images.shape[0], -1)
            
            if args['cuda']:
                images, labels = images.cuda(), labels.cuda()
            
            # Training pass
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            
            loss.backward()

            for name, parameter in model.named_parameters():

                # if name == 'main.0.weight':
                #     gradient = parameter.grad
                #     gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                #     if e >= 1:
                #         for img in img_names:
                #             grad_norm_list1[img, :] += gradient_norm
                            
                # if name == 'main.2.weight':
                #     gradient = parameter.grad
                #     gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                #     if e >= 1:
                #         for img in img_names:
                #             grad_norm_list2[img, :] += gradient_norm
                            
                if name == 'main.4.weight':
                    gradient = parameter.grad
                    gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                    if e >= 1:
                        for img in img_names:
                            grad_norm_list3[img, :] += gradient_norm

            optimizer.step()
            
            training_loss += loss.item()
            
        else:
            train_loss = training_loss / len(trainloader)
            logger.info('epoch {} - train loss: {:.4f}'.format(e+1, train_loss))
            
    
    
    '''
    # ----------------------
    # Test
    # ----------------------
    '''
    correct_count, all_count = 0, 0
    for i, mnist_data in enumerate(testloader):
        img_names = mnist_data['name']
        images = mnist_data['img']
        labels = mnist_data['target']
        
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            if args['cuda']:
                img = img.cuda()
            with torch.no_grad():
                logps = model(img)
    
        ps = torch.exp(logps)
        probab = list(ps.cpu().numpy()[0])
        pred_label = probab.index(max(probab))
        true_label = labels.numpy()[i]
        if(true_label == pred_label):
          correct_count += 1
        all_count += 1
    
    logger.info(f'Number of Images Testes = {all_count}')
    logger.info(f'Model Accuracy(%) = {(correct_count/all_count) * 100}')
    
    
    
    '''
    # ----------------------
    # Save Model & Grad
    # ----------------------
    '''
    if not os.path.exists('model_zoo'):
        os.makedirs('model_zoo')
    model_path = f'model_zoo/mnist_e{epoch}b{batch_size}_{mode}l3.pth'
    torch.save(model.state_dict(), model_path)
    logger.info(f'Saved State Dict in {model_path}')

    if not os.path.exists(f'grad_norm/{mode}'):
        os.makedirs(f'grad_norm/{mode}')
    # norm_path1 = f'grad_norm/{mode}/e{epoch}b{batch_size}l1'
    # norm_path2 = f'grad_norm/{mode}/e{epoch}b{batch_size}l2'
    norm_path3 = f'grad_norm/{mode}/e{epoch}b{batch_size}l3'
    np.save(norm_path3, grad_norm_list3)
    # np.save(norm_path1, grad_norm_list1)
    # np.save(norm_path2, grad_norm_list2)
    # logger.info(f'Saved grad norm Matrix in {norm_path1}')
    # logger.info(f'Saved grad norm Matrix in {norm_path2}')
    





def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    train(args)

    
    
if __name__ == '__main__':
    
    main()



