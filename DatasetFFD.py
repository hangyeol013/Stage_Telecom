#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 11:53:58 2021

@author: kang
"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch

import cv2
import os
import numpy as np
import argparse



# ------------------------------------
# Read and Load images in image list
# ------------------------------------

def _read_image(image_path, is_gray):
    
    if is_gray:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = np.expand_dims(image,0)  # 1xHxW
    else:
        image = cv2.imread(image_path)
    
    if (image.shape[2] == 1) or (image.shape[2] == 3):
        image = image.transpose([2, 0, 1])
    
    return np.float32(image / 255)


def load_images(args, phase, is_gray, base_path):

    if is_gray:
        if phase == 'train':
            path_dir = 'gray/train/Train400/'
        elif phase == 'val':
            path_dir = 'gray/val/BSD68/'
    else:
        if phase == 'train':
            path_dir = 'rgb/train/'
        elif phase == 'val':
            path_dir = 'rgb/val/'
        elif phase == 'test':
            path_dir = f'rgb/{args["test_set"]}/'

    image_dir = base_path + path_dir
    images = []
    for fn in next(os.walk(image_dir))[2]:
        image = _read_image(image_dir + fn, is_gray)
        images.append(image)
    return images


# ------------------------------------
# Images to patches
# ------------------------------------

def _image_to_patches(image, patch_size):
    
    
    H = image.shape[1]
    W = image.shape[2]
    
    if W < patch_size or H < patch_size:
        return []
    
    patches = []
    for patch_h in range(0, H // patch_size):
        for patch_w in range(0, W // patch_size):
            patch = image[:, patch_h*patch_size:(patch_h+1)*patch_size, 
                          patch_w*patch_size:(patch_w+1)*patch_size]
            patches.append(patch)
            
            
    return np.array(patches, dtype = np.float32)


def images_to_patches(images, patch_size):
    patches_list = []
    for image in images:
        # print('image_shape: ', image.shape)
        patches = _image_to_patches(image, patch_size)
        if len(patches) != 0:
            patches_list.append(patches)
            
    images_patches = np.vstack(patches_list)
    
    return images_patches
    


class DatasetFFDNet(Dataset):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        self.patch_size = args['patch_size'] if args['phase']=='train' else 64
        self.phase = args['phase']
        self.base_path = args['base_path']
        self.is_gray = args['is_gray'] if args['phase']=='train' else False
        self.sigma = args['sigma'] if args['phase']=='train' else [0, 75]
        self.sigma_min = self.sigma[0]
        self.sigma_max = self.sigma[1]
        self.sigma_test = args['sigma_test']
        self.H_images = load_images(self.args, self.phase, self.is_gray, self.base_path)
        # self.H_patches = images_to_patches(self.H_images, self.patch_size) if self.phase == 'train' else self.H_images
        self.H_patches = self.H_images if self.phase == 'test' else images_to_patches(self.H_images, self.patch_size)
    def __getitem__(self, index):
        
        H_patch = self.H_patches[index]
        name = index
        
        if self.phase == 'train':
            
            img_H = torch.from_numpy(H_patch)
            img_L = img_H.clone()
            
            noise_level = torch.FloatTensor([np.random.uniform(self.sigma_min, self.sigma_max)])/255.0
            noise = torch.randn(img_L.size()).mul_(noise_level).float()
            img_L.add_(noise)
        
        else:
            img_L = np.copy(H_patch)
            
            img_L += np.random.normal(0, self.sigma_test/255.0, img_L.shape)
            
            noise_level = torch.FloatTensor([self.sigma_test/255.0])
            
            img_H = torch.from_numpy(H_patch).float()
            img_L = torch.from_numpy(img_L).float()
            
            
        return {'name': name, 'L': img_L, 'H': img_H, 'C': noise_level}
    
    def __len__(self):
        return len(self.H_patches)



# def main(json_path = 'Implementation.json'):

#     parser = argparse.ArgumentParser()
#     parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

#     args = parse(parser.parse_args().opt)
    

#     args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    
#     for phase, dataset_opt in args['dataset'].items():
#         if phase == 'train':
#             train_set = DatasetFFDNet(dataset_opt)
#             train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True)
#         elif phase == 'test':
#             test_set = DatasetFFDNet(dataset_opt)
#             test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False)
#         else:
#             raise NotImplementedError("Phase [%s] is not recognized."% phase)
    
#     # dataset = DatasetFFDNet(args)
#     # train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
#     for i, train_data in enumerate(train_dataloader):
#         print(i, train_data['C'].shape)
#     for j, test_data in enumerate(test_dataloader):
#         print(j, test_data['L'].shape)
    

if __name__ == "__main__":
    main()

    
    



