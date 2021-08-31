import torch
import os
import cv2
import numpy as np
import argparse
import random
from collections import OrderedDict
import utils_option
from utils_option import make_logger

from model import FFDNet
from DatasetFFD import DatasetFFDNet



def test(args):

    
    '''
    # ----------------------
    # Seed & Settings
    # ----------------------
    '''
    logger_name = args['test']['logger_name']
    logger_path = args['test']['logger_path']
    logger = make_logger(file_path=logger_path, name=logger_name)
    
    
    seed = args['seed']
    if seed is None:
        seed = random.randint(1, 100)
    logger.info('Random Seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    batch_size = args['method1']['batch_size']
    epoch = args['method1']['epoch']
    is_clip = args['is_clip']
    noise_level_img = args['dataset']['test']['sigma_test']
    
    
    '''
    # ----------------------
    # Dataset
    # ----------------------
    '''
    args_dataset = args['dataset']
    # test_img_idxs = [1,2,3,4,5,6]
    test_set = DatasetFFDNet(args_dataset['test'])
    logger.info(f'test_set: {len(test_set)}')
    
    test_base = args['dataset']['test']['base_path']
    testset_name = args['dataset']['test']['test_set']
    
    if args['is_gray']:
        testset_path = f'gray/{testset_name}/'
    else:
        testset_path = f'rgb/{testset_name}/'
    test_path = os.path.join(test_base, testset_path)
    
    E_path = f'results/{testset_name}'
    if not os.path.exists(E_path):
        os.makedirs(E_path)
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    base_path = 'model_zoo'
    
    if args['is_gray']:
        model_name = f'FFDNet_gray_e{epoch}b{batch_size}.pth'
    else:    
        model_name = f'FFDNet_rgb_e{epoch}b{batch_size}_1000.pth'
    
    model_path = os.path.join(base_path, model_name)
    
    model = FFDNet(is_gray = args['is_gray'])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict = True)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    if args['cuda']:
        model = model.cuda()
        
    test_results =OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    
    result_name = testset_name + '_' + model_name

    logger.info(f'Model: {model_name}, Is_clip: {is_clip}')
    logger.info(f'Noise_level_img: {noise_level_img}')
    

    '''
    # ----------------------
    # Calc Activations
    # ----------------------
    '''
    # for idx in test_img_idxs:
    for idx in range(len(test_set)):
        
        img_name, ext = os.path.splitext(os.listdir(test_path)[idx])
        test_img = test_set[idx]
        img_L = test_img['L']
        img_H = test_img['H']
        img_sigma = test_img['C']
        
        img_L = img_L.unsqueeze(0)
        
        if is_clip:
            img_L = np.float32(np.uint8((img_L.clip(0,1)*255).round())/255.)
            img_L = torch.from_numpy(img_L)
        
        img_L_save = np.uint8((img_L.squeeze()*255).round())
        img_L_save = np.transpose(img_L_save, (1,2,0))

        if args['cuda']:
            img_L, img_H, img_sigma = img_L.cuda(), img_H.cuda(), img_sigma.cuda()
            
        img_E = model(img_L, img_sigma)
        
        img_E = img_E.data.squeeze().float().clamp_(0,1).cpu().numpy()
        img_E = np.transpose(img_E, (1,2,0))
        img_E = np.uint8((img_E * 255).round())
        
        img_H = np.uint8((img_H.squeeze() * 255).round())
        img_H = np.transpose(img_H, (1,2,0))
        
        
        '''
        # --------------------------------
        # Calculate Metrics, Save Images
        # --------------------------------
        '''
        
        psnr = utils_option.calculate_psnr(img_E, img_H, border=args['test']['border'])
        ssim = utils_option.calculate_ssim(img_E, img_H, border=args['test']['border'])
        test_results['psnr'].append(psnr)
        test_results['ssim'].append(ssim)
        logger.info('{} - PSNR: {:.2f} dB;  SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
        img_E_path = f'{img_name}_E_{noise_level_img}.png'
        if is_clip:
            img_L_path = f'{img_name}_L_{noise_level_img}.png'
        else:    
            img_L_path = f'{img_name}_L_noclip_{noise_level_img}.png'
        cv2.imwrite(os.path.join(E_path, img_L_path), img_L_save)
        cv2.imwrite(os.path.join(E_path, img_E_path), img_E)
        
        
    if args['is_gray']:
        task = 'Gray'
    else:
        task = 'RGB'
        
    ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
    ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])

    logger.info('Average PSNR/SSIM ({}) - {} \
                - PSNR: {:.2f} dB, SSIM: {:.4f}'.format(task, result_name, ave_psnr, ave_ssim))
        
        





def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args("").opt)

    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()


    test(args)


if __name__ == '__main__':
    main()

