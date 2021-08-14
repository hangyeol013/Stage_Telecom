#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""


import torch
from torch import nn, optim
from torchvision import transforms
import torchvision
import argparse
import os
import numpy as np
import random

from utils_option import parse, make_logger
from utils_option import topLoss_img, newDataset
from model import mnist_model as net




def train(args):
    
    
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
    method = args['method']
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'method: {method}' )
    logger.info(f'epoch:{epoch}, batch size:{batch_size}')
    
    
    '''
    # ----------------------
    # Datasets
    # ----------------------
    '''
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    # test_img_ids = topLoss_img(args)
    test_img_ids = [3520, 3879, 597, 2667, 5673, 8616, 5570, 9489, 3714, 7362]
    test_set = torchvision.datasets.MNIST(root = './datasets', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
    logger.info(f'{test_img_ids}')
    logger.info('test_set: {}'.format(len(test_set)))
    logger.info(f'DataSet: {args["newData"]}')
    
    
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
    
    
    '''
    # ----------------------
    # expTest Train Start
    # ----------------------
    '''
    logger.info('------> Start Training...')
    vis_num = [200, 500, 1000, 2000, 5000, 10000]
    
    for vis_n in vis_num:
        args['vis_num'] = vis_n
        logger.info(f'change_nums: {args["vis_num"]}')
        for x, test_idx in enumerate(test_img_ids):
            if method == 'method1':
                train_set = newDataset(args=args, test_idx=test_idx, i=x)
            elif method == 'method2':
                train_set = newDataset(args=args, test_idx=test_idx, i=str(x))
            trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
            logger.info(f'test_img: {test_idx}({x+1}/{len(test_img_ids)})')
            
            for e in range(epoch):
                training_loss = 0
                for i, mnist_data in enumerate(trainloader):
                    
                    images = mnist_data[0]
                    labels = mnist_data[1]
                    
                    images = images.view(images.shape[0], -1)
                    
                    if args['cuda']:
                        images, labels = images.cuda(), labels.cuda()
                    
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    
                    loss.backward()
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
                images = mnist_data[0]
                labels = mnist_data[1]
                
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
            # Save expTest Models
            # ----------------------
            '''
            if not os.path.exists(f'model_zoo/expTest/seed{seed}/{args["method"]}_helpful/n{args["vis_num"]}/{args["newData"]}'):
                os.makedirs(f'model_zoo/expTest/seed{seed}/{args["method"]}_helpful/n{args["vis_num"]}/{args["newData"]}')
            model_path = f'model_zoo/expTest/seed{seed}/{args["method"]}_helpful/n{args["vis_num"]}/{args["newData"]}/t{test_idx}.pth'
            torch.save(model.state_dict(), model_path)
            logger.info(f'Saved State Dict in {model_path}')







def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    train(args)

    
    
if __name__ == '__main__':
    
    main()



