import torch
import os
import numpy as np
import argparse
import random
import utils_option
from utils_option import make_logger

from model import FFDNet
from DatasetFFD import DatasetFFDNet
import matplotlib.pyplot as plt



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
    mode = args['method1']['mode']
    #layer: 0 or 5 or 11
    layer = args['method1']['layer']
    layer_n = 3*layer - 4
    if layer_n < 0:
        layer_n = 0
    actv_point = args['method1']['point']
    img_point = [actv_point[0]*2, actv_point[1]*2]
    is_clip = args['is_clip']
    noise_level_img = args['test']['noise_level_img']
    noise_level_model = args['test']['noise_level_model']
    
    
    logger.info(f'cuda: {args["cuda"]}')
    logger.info(f'Is_gray: {args["is_gray"]}, mode: {mode}')
    logger.info(f'epoch:{epoch}, batch size:{batch_size}, layer:{layer}')
    logger.info(f'noise_img: {noise_level_img}, noise_model: {noise_level_model}')
    
    
    '''
    # ----------------------
    # Dataset
    # ----------------------
    '''
    args_dataset = args['dataset']
    test_img_idxs = [args['method1']['vis_num']]
    test_set = DatasetFFDNet(args_dataset['test'])
    logger.info(f'test_set: {len(test_set)}')
    
    test_base = args['dataset']['test']['base_path']
    testset_name = args['dataset']['test']['test_set']
    
    if args['is_gray']:
        testset_path = f'gray/{testset_name}/'
    else:
        testset_path = f'rgb/{testset_name}/'
    test_path = os.path.join(test_base, testset_path)
    
    
    
    '''
    # ----------------------
    # Model & Settings
    # ----------------------
    '''
    if args['is_gray']:
        model_path = f'model_zoo/FFDNet_gray_e{epoch}b{batch_size}_1000.pth'
    else:    
        model_path = f'model_zoo/FFDNet_rgb_e{epoch}b{batch_size}_1000.pth'
    
    model = FFDNet(is_gray = args['is_gray'])
    if args['cuda']:
        model.load_state_dict(torch.load(model_path), strict = True)
        model.cuda()
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict = True)
    model.eval()
    for name, param in model.named_parameters():
        param.requires_grad = False
    
    
    if args['is_clip']:
        is_clip = True
    else:
        is_clip = False

    

    '''
    # ----------------------
    # Calc Activations
    # ----------------------
    '''
    for idx in test_img_idxs:
    # for idx in range(len(test_set)):
        
        img_name, ext = os.path.splitext(os.listdir(test_path)[idx])
        test_img = test_set[idx]
        img_L = test_img['L']
        img_H = test_img['H']
        img_sigma = test_img['C']
        
        img_L = img_L.unsqueeze(0)
        
        if is_clip:
            img_L = np.float32(np.uint8((img_L.clip(0,1)*255).round())/255.)
            img_L = torch.from_numpy(img_L)
        

        if args['cuda']:
            img_L, img_H, img_sigma = img_L.cuda(), img_H.cuda(), img_sigma.cuda()
        
        
        '''
        Check loaded_array, activation Shape
        '''
        
        loaded_array = np.load(f'grad_norm/e{epoch}b{batch_size}l{layer}_1000.npy')
        loaded_sum = np.sum(loaded_array, axis=3)
        loaded_sum = np.sum(loaded_sum, axis=2)

        if args['method1']['mode'] == 'normalized':
            loaded_normalized = loaded_sum/loaded_sum.sum(axis=1).reshape((-1,1))
        else:
            loaded_normalized = loaded_sum

        activation = {}
        def get_activaion(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        model.main[layer_n].register_forward_hook(get_activaion(f'{layer_n}'))
        output = model(img_L, img_sigma)
        activation_val = activation[f'{layer_n}'].cpu().numpy()
        actv_val = activation_val[:,:,actv_point[0], actv_point[1]]
        # actv_val = activation_val[:, :, 
        #                           actv_point[0]-1:actv_point[0]+2, 
        #                           actv_point[1]-1:actv_point[1]+2]
        actv_val[actv_val<0] = 0
        
        mul_list = []
        for i in range(loaded_array.shape[0]):
            vdot_product = np.vdot(loaded_normalized[i,:], actv_val[0,:])
            # act_prod = np.einsum('k,k', loaded_array[i,:], actv_val[0,:])
            # act_prod = np.einsum('kij,kij',loaded_array[i,:,:,:],actv_val[0,:,:,:])
            # mul_list.append(act_prod)
            mul_list.append(vdot_product)
            
        mul_array = np.asarray(mul_list)

        '''
        # ----------------------
        # Save Activations
        # ----------------------
        '''
        if not os.path.exists(f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}'):
            os.makedirs(f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}')
        
        norm_path = f'act_grad/{mode}/layer{layer}/{testset_name}/{img_name}/{img_point}'
        np.save(norm_path, mul_array)
        logger.info(f'Saved act_grad values in {norm_path}')

        # Only activations (when you want to save only activations not the multiplication of grad_norm and activations)
        # When you run here, make the code (from 159 to 179) into comments
        # act_path = f'activations/{img_name}_l{layer}_{img_point}'
        # if not os.path.exists('activations'):
        #     os.makedirs('activations')
        # np.save(act_path, actv_val)





def main(json_path = 'Implementation.json'):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args("").opt)

    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()


    test(args)


if __name__ == '__main__':
    main()

