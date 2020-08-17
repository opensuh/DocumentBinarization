import os
import numpy as np
import torch
import cv2
import argparse

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import sys
sys.path.append('../Common')
from tool_clean import get_image_patch, check_is_image

# make step1 prediction image patch, and test image for step2 training.
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='1', help="GPU number")
parser.add_argument("--fold_num", type=int, default=0, help="fold number, 0 ~ fold_total-1")
parser.add_argument("--fold_total", type=int, default=5, help="fold total")

parser.add_argument('--base_model_name', type=str, default='efficientnet-b4', help='base model name')
parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='none or imagenet')
parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
parser.add_argument('--threshold', type=float, default=0.30, help='threshold for bgr mask')


parser.add_argument('--original_dir', type=str, default='/mnt/nas/data/denoise/LRDE/', help='original image dir')

opt = parser.parse_args()

device = torch.device("cuda:%s" % opt.gpu)

models = []
base_model_name = opt.base_model_name
lambda_bce = opt.lambda_bce
generator_lr = opt.generator_lr
encoder_weights = opt.encoder_weights
thresh_hold = opt.threshold

weight_folder = ('./step1_LRDE%d_' % opt.fold_num) + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(thresh_hold) + '/'
weight_list = sorted(os.listdir(weight_folder))
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list 
                    if weight_path.endswith('pth') and 'unet' in weight_path]
print(weight_list)

# blue
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# green
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[1], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# red
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[2], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# gray
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[3], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

batch_size = 16
preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

# make directory
image_save_path = './predicted_image_for_step2_lrde_%d' % opt.fold_num
os.makedirs(image_save_path, exist_ok=True)

train_image_save_path = os.path.join(image_save_path, 'train')
os.makedirs(train_image_save_path, exist_ok=True)

test_image_save_path = os.path.join(image_save_path, 'test')
os.makedirs(test_image_save_path, exist_ok=True)

# patch directory
patch_save_path = os.path.join(train_image_save_path, 'patch')
os.makedirs(patch_save_path, exist_ok=True)

patch_train_image_save_path = os.path.join(patch_save_path, 'image')
os.makedirs(patch_train_image_save_path, exist_ok=True)

patch_train_mask_save_path = os.path.join(patch_save_path, 'mask')
os.makedirs(patch_train_mask_save_path, exist_ok=True)
# no test patch
# end patch

# end directory

# start train
step2_overlap_ratio = 0.3
reshape = (256, 256)

predict_overlap_ratio = 0.5
crop_h = 256
crop_w = 256

# get test image according to fold_num
fold_num = opt.fold_num
fold_total = opt.fold_total

original_dir = opt.original_dir
original_image_name = []
origianl_image_pathes = os.listdir(os.path.join(original_dir, 'image'))
origianl_image_pathes = sorted(origianl_image_pathes)
for origianl_image_path in origianl_image_pathes:
    if not check_is_image(origianl_image_path):
        print('not image', origianl_image_path)
        continue
    original_image_name.append(origianl_image_path.split('.')[0])

test_image_name = original_image_name[fold_num::fold_total]
train_image_name = [i for i in original_image_name if i not in test_image_name]
# end test image

image_dir = opt.original_dir
root_image_path = os.path.join(image_dir, 'image')
root_mask_path = os.path.join(image_dir, 'mask')
image_path_list = os.listdir(root_image_path)
for image_path in image_path_list:
    if not check_is_image(image_path):
        print('not image', image_path)
        continue

    image_name = image_path.split('.')[0]
    print('processing the image:', image_name)

    image = cv2.imread(os.path.join(root_image_path, image_path))

    # find and read mask file
    if not os.path.isfile(os.path.join(root_mask_path, image_path)):
        print(image_path, 'no mask')
        exit(1)

    mask = cv2.imread(os.path.join(root_mask_path, image_path), cv2.IMREAD_GRAYSCALE)
    mask[mask <= 128] = 0
    mask[mask > 128] = 255

    h, w, _ = image.shape
    image_patches, poslist = get_image_patch(image, crop_h, crop_w, overlap=predict_overlap_ratio, is_mask=False)

    merge_img = np.ones((h, w, 3))
    out_imgs = []

    for channel in range(4):
        color_patches = []
        for patch in image_patches:
            tmp = patch.astype(np.float32)
            if channel != 3:
                color_patches.append(preprocess_input(tmp[:, :, channel:channel+1]))
            else:
                color_patches.append(preprocess_input(np.expand_dims( cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY), axis=-1 )))

        step = 0
        preds = []
        with torch.no_grad():
            while step < len(image_patches):
                ps = step
                pe = step + batch_size
                if pe >= len(image_patches):
                    pe = len(image_patches)

                target = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float()
                preds.extend(torch.sigmoid(models[channel](target.to(device))).cpu())
                step += batch_size

        # handling overlap
        out_img = np.ones((h, w, 1)) * 255
        for i in range(len(image_patches)):
            patch = preds[i].permute(1, 2, 0).numpy() * 255

            start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
            h_cut = end_h - start_h
            w_cut = end_w - start_w

            tmp = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])
            out_img[start_h:end_h, start_w:end_w] = tmp
        out_imgs.append(out_img)

    # save step1 merged color train image
    merge_img[:, :, 0:1] = (out_imgs[0] + out_imgs[3]) / 2.
    merge_img[:, :, 1:2] = (out_imgs[1] + out_imgs[3]) / 2.
    merge_img[:, :, 2:3] = (out_imgs[2] + out_imgs[3]) / 2.
    merge_img = merge_img.astype(np.uint8)

    is_test = False
    if image_name in test_image_name:
        is_test = True

    cv2.imwrite('%s/%s.png' % (test_image_save_path if is_test else train_image_save_path, image_name), merge_img)

    if is_test:
        continue

    # start patch
    # (patches, 256, 256, 3)
    image_patches, poslist = get_image_patch(merge_img, crop_h, crop_w, step2_overlap_ratio, False)
    mask_patches, poslist = get_image_patch(mask, crop_h, crop_w, step2_overlap_ratio, True)

    for idx in range(len(image_patches)):
        image_patch = image_patches[idx]
        mask_patch = mask_patches[idx]

        img_color_tmp = image_patch
        mask_gray_tmp = mask_patch
        cv2.imwrite('%s/%s_i%dh0.png' % (patch_train_image_save_path, image_name, idx), img_color_tmp)
        cv2.imwrite('%s/%s_i%dh0.png' % (patch_train_mask_save_path, image_name, idx), mask_gray_tmp)

        # horizontal axis
        img_color_tmp = np.flipud(image_patch)
        mask_gray_tmp = np.flipud(mask_patch)
        cv2.imwrite('%s/%s_i%dh1.png' % (patch_train_image_save_path, image_name, idx), img_color_tmp)
        cv2.imwrite('%s/%s_i%dh1.png' % (patch_train_mask_save_path, image_name, idx), mask_gray_tmp)

    # exit(1)
    # end patch
