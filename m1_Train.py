import numpy as np
import argparse
import random
import os

import torch
from torch.utils.data import DataLoader

from model import FFDNet
from DatasetFFD import DatasetFFDNet
import utils_option
from utils_option import select_optimizer
from utils_option import select_lossfn
from utils_option import make_logger

from numpy.linalg import norm



'''
# --------------------------------------
# Training FFDNet
# --------------------------------------
'''
def train(args):
    
    
    '''
    # ----------------------------------------
    # Seed & Settings
    # ----------------------------------------
    '''
    logger_name = args['train']['logger_name']
    logger_path = args['train']['logger_path']
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
    # layer = args['method1']['layer']
    # layer_n = 3*layer - 4
    # if layer_n < 0:
    #     layer_n = 0
    layer_0 = 0
    layer_5 = 11
    layer_11 = 29
    mode = args['method1']['mode']
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'Is_gray: {args["is_gray"]}, mode: {mode}')
    logger.info(f'epoch:{epoch}, batch size:{batch_size}, layer: 0,5,11')
    # logger.info(f'epoch:{epoch}, batch size:{batch_size}, layer:{layer}')
    
    

    '''
    # ----------------------------------------
    # DataLoader
    # ----------------------------------------
    '''
    for phase, dataset_opt in args['dataset'].items():
        if phase == 'train':
            train_set = DatasetFFDNet(dataset_opt)
            train_dataloader = DataLoader(train_set, 
                                          batch_size=batch_size, num_workers=dataset_opt['num_workers'],
                                          shuffle=True)  
            
    logger.info(f'train_set: {len(train_set)}')
    
    
    '''
    # ----------------------------------------
    # Model & Settings
    # ----------------------------------------
    '''
    model = FFDNet(is_gray=args['is_gray'])
    model.apply(utils_option.init_weights)
    if args['cuda']:
        model = model.cuda()
    loss_fn = select_lossfn(opt=args['train']['loss_fn'], reduction=args['train']['reduction'])
    optimizer = select_optimizer(opt=args['train']['optimizer'], lr=args['train']['learning_rate'], model=model)

    grad_norm_0 = np.zeros([len(train_set), 96, 3, 3])
    grad_norm_5 = np.zeros([len(train_set), 96, 3, 3])
    grad_norm_11 = np.zeros([len(train_set), 96, 3, 3])
    
    
    '''
    # ----------------------
    # Training & Calc Grad
    # ----------------------
    '''
    logger.info('------> Start training...')
    for epoch_idx in range(epoch):
        logger.info(f'Epoch: {epoch_idx+1}/{epoch}')
        
        loss_idx = 0
        train_losses = 0
        model.train()

        for batch_idx, batch_data in enumerate(train_dataloader):
            
            img_name = batch_data['name']
            train_batch = batch_data['L']
            label_batch = batch_data['H']
            sigma = batch_data['C']
            
            if args['cuda']:
                train_batch, label_batch, sigma = train_batch.cuda(), label_batch.cuda(), sigma.cuda()

            
            output_batch = model(train_batch, sigma)
            train_loss = loss_fn(output_batch, label_batch)
            train_losses += train_loss
            loss_idx += 1
            
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            
            '''
            # -----------------------------------------------
            # For v 
            # -----------------------------------------------
            '''
            for name, parameter in model.named_parameters():
                
                if name == f'main.{layer_0}.weight':
                    
                    gradient = parameter.grad
                    gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                    
                    for img in img_name:
                        grad_norm_0[img, :, :, :] += gradient_norm
                        
                if name == f'main.{layer_5}.weight':
                    
                    gradient = parameter.grad
                    gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                    
                    for img in img_name:
                        grad_norm_5[img, :, :, :] += gradient_norm
                        
                if name == f'main.{layer_11}.weight':
                    
                    gradient = parameter.grad
                    gradient_norm = norm(gradient.cpu(), ord=2, axis=1)
                    
                    for img in img_name:
                        grad_norm_11[img, :, :, :] += gradient_norm
            
        train_losses /= loss_idx
        logger.info(f', Avg_Train_Loss: {train_losses}')




    '''
    # --------------------------
    # Final Save Model Dict
    # --------------------------
    '''
    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}_1000_0807.pth'
    else:    
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}_1000_0807.pth'
    torch.save(model.state_dict(), model_path)
    
    
    if not os.path.exists('grad_norm'):
        os.makedirs('grad_norm')
        
    norm_path_0 = f'grad_norm/e{epoch}b{batch_size}l0_1000_0807'
    np.save(norm_path_0, grad_norm_0)
    norm_path_5 = f'grad_norm/e{epoch}b{batch_size}l5_1000_0807'
    np.save(norm_path_5, grad_norm_5)
    norm_path_11 = f'grad_norm/e{epoch}b{batch_size}l11_1000_0807'
    np.save(norm_path_11, grad_norm_11)
    
    
    logger.info(f'Saved State Dict in {model_path}')
    logger.info(f'Saved Grad Norm in {norm_path_0}')
    logger.info(f'Saved Grad Norm in {norm_path_5}')
    logger.info(f'Saved Grad Norm in {norm_path_11}')
    
    
    
                



def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args("").opt)
    
    assert args['is_train']

    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    
    
    train(args)


if __name__ == "__main__":
    main()



