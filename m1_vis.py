#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 20 11:41:30 2021

@author: kang
"""

import torch

import matplotlib.pyplot as plt
import argparse
import numpy as np
import os


from DatasetFFD import DatasetFFDNet
from model import FFDNet
from utils_option import parse, make_logger
import utils_option as utils

    


def visual(args):
    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['test']['logger_name']
    logger_path = args['test']['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    
    '''
    # ---------------------------
    # Dataset
    # ---------------------------
    # '''
    args_dataset = args['dataset']
    train_set = DatasetFFDNet(args_dataset['train'])
    test_set = DatasetFFDNet(args_dataset['test'])
        
    test_img_num = args['method1']['vis_num']
    test_img = test_set[test_img_num]
    
    test_L = test_img['L']
    test_H = test_img['H']
    test_C = test_img['C']
    
    test_L_view = np.uint8((test_L.squeeze()*255).round())
    test_L_view = np.transpose(test_L_view, (1,2,0))
    test_L_view = test_L_view[:, :, [2,1,0]]
    
    test_H_view = np.uint8((test_H.squeeze()*255).round())
    test_H_view = np.transpose(test_H_view, (1,2,0))
    test_H_view = test_H_view[:, :, [2,1,0]]
    
    test_base = args['dataset']['test']['base_path']
    testset_name = args['dataset']['test']['test_set']
    
    if args['is_gray']:
        testset_path = f'gray/{testset_name}/'
    else:
        testset_path = f'rgb/{testset_name}/'
    test_path = os.path.join(test_base, testset_path)
    img_name, _ = os.path.splitext(os.listdir(test_path)[test_img_num])
    
    
    
    
    '''
    # ---------------------------
    # Settings
    # ---------------------------
    '''
    epoch = args['method1']['epoch']
    batch_size = args['method1']['batch_size']
    mode = args['method1']['mode']
    layer = args['method1']['layer']
    actv_point = args['method1']['point']  
    img_point = [actv_point[0]*2, actv_point[1]*2]
    img_point1 = [img_point[0], img_point[1]*2]
    img_point2 = [img_point[0], img_point[1]*3]
    img_point3 = [img_point[0], img_point[1]*4]
    is_clip = args['is_clip']
    
    logger.info(f'epoch:{epoch}, batch size:{batch_size}, is_clip:{is_clip}')
    
    
    
    '''
    # ---------------------------
    # Model & Predict results
    # ---------------------------
    '''
    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}.pth'
    else:    
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}_1000.pth'
    
    model = FFDNet(is_gray=args['is_gray'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=True)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
        
    test_L = test_L.unsqueeze(0)
    if is_clip:
        test_L = np.float32(np.uint8((test_L.clip(0,1)*255).round())/255.)
        test_L = torch.from_numpy(test_L)
    
    test_P = model(test_L, test_C)
    test_P_view = np.uint8((test_P.squeeze()*255).round())
    test_P_view = np.transpose(test_P_view, (1,2,0))
    test_P_view = test_P_view[:, :, [2,1,0]]
    
    logger.info(f'img_name: {img_name}')
    args['base_path'] = f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}/{img_point}.npy'
    act_grad = np.load(args['base_path'])
    top_images = utils.topN_images_list(args=args)[:5]
    
    args['base_path'] = f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}/{img_point1}.npy'
    act_grad1 = np.load(args['base_path'])
    top_images1 = utils.topN_images_list(args=args)[:5]
    
    args['base_path'] = f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}/{img_point2}.npy'
    act_grad2 = np.load(args['base_path'])
    top_images2 = utils.topN_images_list(args=args)[:5]
    
    args['base_path'] = f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}/{img_point3}.npy'
    act_grad3 = np.load(args['base_path'])
    top_images3 = utils.topN_images_list(args=args)[:5]
    
    
    
    
    '''
    # ---------------------------
    # Plot images
    # ---------------------------
    '''
    plt.clf()
    for i, (img_num, img_num1, img_num2, img_num3) in enumerate(zip(top_images, top_images1, top_images2, top_images3)):

        top_train = train_set[img_num]
        train_H = top_train['H']
        train_H_view = np.uint8((train_H.squeeze()*255).round())
        train_H_view = np.transpose(train_H_view, (1,2,0))
        train_H_view = train_H_view[:, :, [2,1,0]]
        
        act_grad_v = act_grad[img_num]
        
        
        top_train1 = train_set[img_num1]
        train_H1 = top_train1['H']
        train_H1_view = np.uint8((train_H1.squeeze()*255).round())
        train_H1_view = np.transpose(train_H1_view, (1,2,0))
        train_H1_view = train_H1_view[:, :, [2,1,0]]
        
        act_grad_v1 = act_grad1[img_num1]
        
        
        top_train2 = train_set[img_num2]
        train_H2 = top_train2['H']
        train_H2_view = np.uint8((train_H2.squeeze()*255).round())
        train_H2_view = np.transpose(train_H2_view, (1,2,0))
        train_H2_view = train_H2_view[:, :, [2,1,0]]
        
        act_grad_v2 = act_grad2[img_num2]
        
        
        top_train3 = train_set[img_num3]
        train_H3 = top_train3['H']
        train_H3_view = np.uint8((train_H3.squeeze()*255).round())
        train_H3_view = np.transpose(train_H3_view, (1,2,0))
        train_H3_view = train_H3_view[:, :, [2,1,0]]
        
        act_grad_v3 = act_grad3[img_num3]
        
        
        plt.suptitle(f'Top 5 images of {img_name}.png (layer: {layer}, point: {img_point[0]})')
        plt.subplot(4,6,1)
        plt.title(f'test_id: {test_img_num}\npoint: {img_point}', fontdict={'fontsize':10}, y=-0.3)
        plt.scatter([img_point[0]], [img_point[1]], c='r', s=2)
        plt.axis('off')
        plt.imshow(test_H_view)
        
        plt.subplot(4,6,i+2)
        plt.title('train_idx: {}\nvalue: {:.2f}'.format(img_num, act_grad_v), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_H_view)
        
        plt.subplot(4,6,7)
        plt.title(f'test_id: {test_img_num}\npoint: {img_point1}', fontdict={'fontsize':10}, y=-0.3)
        plt.scatter([img_point1[0]], [img_point1[1]], c='r', s=2)
        plt.axis('off')
        plt.imshow(test_H_view)
        
        plt.subplot(4,6,i+8)
        plt.title('train_id: {}\nvalue: {:.2f}'.format(img_num1, act_grad_v1), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_H1_view)
        
        plt.subplot(4,6,13)
        plt.title(f'test_id: {test_img_num}\npoint: {img_point2}', fontdict={'fontsize':10}, y=-0.3)
        plt.scatter([img_point2[0]], [img_point2[1]], c='r', s=2)
        plt.axis('off')
        plt.imshow(test_H_view)
        
        plt.subplot(4,6,i+14)
        plt.title('train_id: {}\nvalue: {:.2f}'.format(img_num2, act_grad_v2), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_H2_view)
        
        plt.subplot(4,6,19)
        plt.title(f'test_id: {test_img_num}\npoint: {img_point3}', fontdict={'fontsize':10}, y=-0.3)
        plt.scatter([img_point3[0]], [img_point3[1]], c='r', s=2)
        plt.axis('off')
        plt.imshow(test_H_view)
        
        plt.subplot(4,6,i+20)
        plt.title('train_id: {}\nvalue: {:.2f}'.format(img_num3, act_grad_v3), 
                  fontdict={'fontsize':9}, y=-0.3)
        plt.axis('off')
        plt.imshow(train_H3_view)
        
        
        
        
    if not os.path.exists(f'fig/m1/{mode}/{testset_name}'):
        os.makedirs(f'fig/m1/{mode}/{testset_name}')
    if args['method1']['remove_out']:
        fig_path = f'fig/m1/{mode}/{testset_name}/{img_name}_{layer}_{img_point[0]}_out{args["method1"]["outlim"]}.png'
    else:
        fig_path = f'fig/m1/{mode}/{testset_name}/{img_name}_{layer}_{img_point[0]}_withOutlier.png'
    plt.savefig(fig_path)






def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = parse(parser.parse_args("").opt)
    
    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    args['vis'] = True

    visual(args)
        
    
    
if __name__ == '__main__':
    
    main()



