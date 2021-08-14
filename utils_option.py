import cv2
import numpy as np
import math
import logging
import json
import torch
import torch.nn as nn
import sys
from skimage.util import random_noise
import torch.optim as optim
from DatasetFFD import DatasetFFDNet
from model import FFDNet


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
    
    epoch = args['method2']['epoch']
    batch_size = args['method2']['batch_size']
    
    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}.pth'
    else:    
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}_1000.pth'
    
    net = FFDNet(is_gray=args['is_gray'])
    if args['cuda']:
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    return net


'''
# -------------------------
# Load Data (for IF)
# -------------------------
'''
def load_data(args):
    
    args_dataset = args['dataset']
    train_set = DatasetFFDNet(args_dataset['train'])
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=32, num_workers=4, shuffle=True)
    
    test_set = DatasetFFDNet(args_dataset['test'])
    testloader = torch.utils.data.DataLoader(test_set, batch_size=1, num_workers=4, shuffle=False)
    

    return trainloader, testloader


'''
# ----------------------------------
# Orthogonal Initialization
# ----------------------------------
'''
def init_weights(model):
    for i in model.children():
        if isinstance(i, nn.Conv2d):
            nn.init.orthogonal_(i.weight.data)

'''
# ----------------------------------
# Loss function
# ----------------------------------
'''
def select_lossfn(opt, reduction='mean'):
    if opt == 'l1':
        lossfn = nn.L1Loss(reduction=reduction)
    elif opt == 'l2':
        lossfn = nn.MSELoss(reduction=reduction)
    return lossfn

'''
# ----------------------------------
# Optimizer
# ----------------------------------
'''
def select_optimizer(opt, lr, model):
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    return optimizer

'''
# ----------------------------------
# Adding noise to image batch (return: n * C * H * W)
# ----------------------------------
'''
def add_batch_noise(images, noise_sigma):

    images = random_noise(images.numpy(), mode = 'gaussian', var = noise_sigma)
    return torch.FloatTensor(images)


'''
# ----------------------------
# Calculate PSNR
# ----------------------------
'''
def calculate_psnr(img1, img2, border = 0):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


'''
# ----------------------------
# Calculate SSIM
# ----------------------------
'''
def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, border = 0):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:, :, i], img2[:, :, i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions')

    
'''
# -------------------------
# TopN images list
# -------------------------
'''
def topN_images_list(args):
    
    base_path = args['base_path']
    loaded_v = np.load(base_path)
    v = loaded_v
    top_images = sorted(range(len(v)), key = lambda k: v[k], reverse=True)
    
    if args['method1']['remove_out']:
        outliers = FindOutlier(args)
        top_images = [top_image for top_image in top_images if top_image not in outliers][:100]
    else:
        top_images = top_images[:100]
    
    
    return top_images




'''
# -------------------------
# Top Loss img list
# -------------------------
'''
def topLoss_img(args):
    
    args_dataset = args['dataset']
    test_set = DatasetFFDNet(args_dataset['test'])
    
    epoch = args['method2']['epoch']
    batch_size = args['method2']['batch_size']
    is_clip = args['is_clip']

    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}.pth'
    else:    
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}.pth'
    
    net = FFDNet(is_gray=args['is_gray'])
    if args['cuda']:
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
    loss_fn = select_lossfn(opt=args['train']['loss_fn'], reduction=args['train']['reduction'])
    
    losses = []
    
    for idx in range(len(test_set)):
        
        test_img = test_set[idx]
        img_L = test_img['L']
        img_H = test_img['H']
        img_C = test_img['C']
        
        img_L = img_L.unsqueeze(0)
       
        if is_clip:
            img_L = np.float32(np.uint8((img_L.clip(0,1)*255).round())/255.)
            img_L = torch.from_numpy(img_L)
        
        if args['cuda']:
            img_L, img_H, img_C = img_L.cuda(), img_H.cuda(), img_C.cuda()
            
        img_O = net(img_L, img_C)
        loss = loss_fn(img_O, img_H)
        
        losses.append(loss.item())
        
    top_loss = sorted(range(len(losses)), key=lambda k:losses[k], 
                      reverse=True)

    return top_loss




'''
# -------------------------
# Find Outlier
# -------------------------
'''
def FindOutlier(args):

    loaded_array = np.load(f'grad_norm/e{args["method1"]["epoch"]}b{args["method1"]["batch_size"]}l{args["method1"]["layer"]}_1000.npy')
    loaded_sum = np.sum(loaded_array, axis=3)
    loaded_sum = np.sum(loaded_sum, axis=2)
    gn_sum = np.sum(loaded_sum, axis=1)
    
    x = [k for k in range(len(gn_sum)) if gn_sum[k] > args['method1']['outlim']]
    
    # imgs = [25917, 38963, 10294, 6953, 20860, 5173, 36434, 45293, 10806, 19406, 
    #         19622, 6778, 45510, 39891, 34947]
    # imgs1 = [25917, 38963, 10294, 6953, 20860]
    # imgs_set = set(imgs)
    # int01 = imgs_set.intersection(x)
    # int02 = imgs_set.intersection(imgs1)
    # print(int01)
    # print(int02)
    # print('imgs_len: ', len(imgs))
    # print('int_len: ', len(int01))
    
    return x
    
    