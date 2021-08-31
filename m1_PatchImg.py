import argparse
import torch
import os
from PIL import Image

from DatasetFFD import load_images
import utils_option
import utils_option as utils





def _image_to_patch_count(image, patch_size):
    H = image.shape[1]
    W = image.shape[2]
    
    if W < patch_size or H < patch_size:
        return 0
    
    count = 0
    for patch_h in range(0, H // patch_size):
        for patch_w in range(0, W // patch_size):
            count += 1
            
    return count


def images_to_patches_count(images, patch_size):
    count_list = []
    for image in images:
        count = _image_to_patch_count(image, patch_size)
        count_list.append(count)
            
    return count_list




'''
# --------------------------------------
# Training FFDNet
# --------------------------------------
'''
def trace_img(args):
    
    
    
    args_test = args['dataset']['test']
    patch_size = args['dataset']['train']['patch_size']
    base_path = args['dataset']['train']['base_path']
    
    H_images = load_images(args=args_test, phase='train', is_gray=False, base_path=base_path)
    patches_count = images_to_patches_count(images=H_images, patch_size=patch_size)
    print('H_images: ', len(H_images))
    # print('H_patches: ', patches_count)
    
    mode = args['method1']['mode']
    layer = args['method1']['layer']
    actv_point = args['method1']['point']

    test_img_num = args['method1']['vis_num']
    test_base = args['dataset']['test']['base_path']
    testset_name = args['dataset']['test']['test_set']
    
    if args['is_gray']:
        testset_path = f'gray/{testset_name}/'
    else:
        testset_path = f'rgb/{testset_name}/'
    test_path = os.path.join(test_base, testset_path)
    img_name, _ = os.path.splitext(os.listdir(test_path)[test_img_num])
    print('test_img: ', img_name)
    args['base_path'] = f'act_grad/{mode}/layer{layer}/{testset_name}/{actv_point}/{img_name}.npy'
    top_patches = utils.topN_images_list(args=args)
    
    print('top_patches: ', top_patches)
    
    top_images = []
    for img_num in top_patches:
        for i, img in enumerate(patches_count):
            img_num -= img
            if img_num <= 0:
                top_images.append(i)
                break
        
    print('top_images: ', top_images)

    train_base = args['dataset']['train']['base_path']
    trainset_path = 'rgb/train'
    train_path = os.path.join(train_base, trainset_path)
    
    train_imgs = os.listdir(train_path)
    

    print(train_imgs[top_images[0]])
    print(train_imgs[top_images[1]])
    print(train_imgs[top_images[2]])
    print(train_imgs[top_images[3]])
    print(train_imgs[top_images[4]])
    
    
    im = Image.open(os.path.join(train_path, train_imgs[top_images[0]]))
    im.show()
    
                



def main(json_path = 'Implementation.json'):

    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, default=json_path, help='Path to option JSON file.')

    args = utils_option.parse(parser.parse_args("").opt)

    args['cuda'] = args['use_gpu'] and torch.cuda.is_available()
    
    
    trace_img(args)


if __name__ == "__main__":
    main()



