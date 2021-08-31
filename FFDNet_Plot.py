import numpy as np
import argparse
import random

import torch
from torch.utils.data import DataLoader

from model import FFDNet
from DatasetFFD import DatasetFFDNet
import utils_option
from utils_option import select_optimizer
from utils_option import select_lossfn
from utils_option import make_logger

import matplotlib.pyplot as plt



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

    batch_size = args['grad']['batch_size']
    epoch = args['grad']['epoch']
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'Is_gray: {args["is_gray"]}')
    logger.info(f'epoch:{epoch}, batch size:{batch_size}')


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
        elif phase == 'val':
            val_set = DatasetFFDNet(dataset_opt)
            val_dataloader = DataLoader(val_set,
                                        batch_size=batch_size, num_workers=dataset_opt['num_workers'],
                                        shuffle=True)
            
    logger.info(f'train_set: {len(train_set)}')
    logger.info(f'val_set: {len(val_set)}')
    
    
    model = FFDNet(is_gray=args['is_gray'])
    model.apply(utils_option.init_weights)
    if args['cuda']:
        model = model.cuda()
    loss_fn = select_lossfn(opt=args['train']['loss_fn'], reduction=args['train']['reduction'])
    optimizer = select_optimizer(opt=args['train']['optimizer'], lr=args['train']['learning_rate'], model=model)
    
    
    train_loss_list = []
    val_loss_list = []
    
    logger.info('> Start training...')
    for epoch_idx in range(args['train']['epochs']):
        logger.info(f'Epoch: {epoch_idx+1}/{args["train"]["epochs"]}')
        
        loss_idx = 0
        train_losses = 0
        model.train()

        for batch_idx, batch_data in enumerate(train_dataloader):
            
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
            
            
        train_losses /= loss_idx
        logger.info(f', Avg_Train_Loss: {train_losses}')
        train_loss_list.append(train_losses)
        

        # ----------------
        # Eveluate
        # ----------------
        loss_idx = 0
        val_losses = 0
        if (epoch_idx + 1) % args['train']['val_epoch'] != 0:
            continue
        model.eval()

        for batch_idx, batch_data in enumerate(val_dataloader):
            
            test_batch = batch_data['L']
            label_batch = batch_data['H']
            sigma = batch_data['C']
            
            if args['cuda']:
                test_batch, label_batch, sigma = test_batch.cuda(), label_batch.cuda(), sigma.cuda()
            
            with torch.no_grad():    
                output_batch = model(test_batch, sigma)
            val_loss = loss_fn(output_batch, label_batch)
            val_losses += val_loss
            loss_idx += 1
                
        val_losses /= loss_idx
        logger.info(f', Avg_Val_Loss: {val_losses}')
        val_loss_list.append(val_losses)
        

    
    
    plt.figure(figsize=(10,5))
    plt.title("Training and Validation Loss")
    plt.plot(val_loss_list,label="val_loss")
    plt.plot(train_loss_list,label="train_loss")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
                



def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args("").opt)

    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()

    train(args)


if __name__ == "__main__":
    main()



