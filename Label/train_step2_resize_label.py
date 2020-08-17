from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.autograd as autograd
import torch
import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randrange
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from custom_dataset_label import Dataset_Return_One

import sys
sys.path.append('../Common')
from model import Discriminator
from tool_clean import get_image_patch, check_is_image, image_padding

def sample_images(epoch, idx, test_images, test_masks, test_masks_pred, image_save_path):
    r, c = 3, 3
    gen_imgs = []
    gen_imgs.extend(test_images)
    gen_imgs.extend(test_masks_pred)
    gen_imgs.extend(test_masks)

    titles = ['Image', 'Gen', 'GT']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if len(gen_imgs[cnt].shape) > 2:
                axs[i, j].imshow(gen_imgs[cnt])
            else:
                axs[i, j].imshow(gen_imgs[cnt], cmap='gray', vmin=0, vmax=1.0)

            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig('%s/%d_%d.png' % (image_save_path, epoch, idx))
    plt.close()


def compute_gradient_penalty(D, real_samples, fake_samples, device):
    # Random weight term for interpolation between real and fake samples
    alpha = torch.FloatTensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0] 
    gradients = gradients.view(gradients.size(0), -1) + 1e-16
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def unet_train(epochs, gpu, base_model_name, encoder_weights, generator_lr, discriminator_lr, lambda_bce,
                batch_size, image_train_dir, mask_train_dir, original_dir, fold_num, fold_total):
    # rgb , preprocess input
    imagenet_mean = np.array( [0.485, 0.456, 0.406] )
    imagenet_std = np.array( [0.229, 0.224, 0.225] )

    train_data_set = Dataset_Return_One(image_train_dir, mask_train_dir, os.path.join(original_dir, 'image'),
                                            base_model_name, encoder_weights, is_train=True, fold_num=fold_num, fold_total=fold_total)

    train_loader = DataLoader(train_data_set, batch_size=batch_size, num_workers=4, shuffle=True)

    weight_path = ('./step2_resize_label_%d_' % fold_num) + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr)
    image_save_path = weight_path + '/images'
    os.makedirs(weight_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)

    device = torch.device("cuda:%s" % gpu)

    # step2 resize unet
    model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
    model.to(device)
    optimizer_generator = optim.Adam(model.parameters(), lr=generator_lr, betas=(0.5, 0.999))

    discriminator = Discriminator(in_channels=4)
    discriminator.to(device)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()

    image_test_list = []
    test_image_pathes = os.listdir(os.path.join(original_dir, 'image'))
    test_image_pathes = sorted(test_image_pathes)
    for test_image_path in test_image_pathes:
        if not check_is_image(test_image_path):
            print('not image', test_image_path)
            continue
        image_test_list.append( (os.path.join(original_dir, 'image', test_image_path), 
                                    os.path.join(original_dir, 'mask', test_image_path)) )
    image_test_list = image_test_list[fold_num::fold_total]
    print('test len:', len(image_test_list))

    preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

    value = int(256 * 0.5)
    lambda_gp = 10.0
    reshape = (512, 512)
    skip_resize_ratio = 6
    skip_max_length = 512
    padding_resize_ratio = 4
    kernel = np.ones((5, 5), np.uint8)

    best_fmeasure = 0.0
    epoch_start_time = time.time()
    for epoch in range(epochs):

        # train
        model.train()
        for idx, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            masks_pred = model(images)

            # discriminator
            discriminator.requires_grad_(True)
            # Fake
            fake_AB = torch.cat((images, masks_pred), 1).detach()
            pred_fake = discriminator(fake_AB)

            # Real
            real_AB = torch.cat((images, masks), 1)
            pred_real = discriminator(real_AB)

            gradient_penalty = compute_gradient_penalty(discriminator, real_AB, fake_AB, device)
            discriminator_loss = -torch.mean(pred_real) + torch.mean(pred_fake) + lambda_gp * gradient_penalty

            optimizer_discriminator.zero_grad()
            discriminator_loss.backward()
            optimizer_discriminator.step()

            if idx % 5 == 0:
                discriminator.requires_grad_(False)

                # generator
                fake_AB = torch.cat((images, masks_pred), 1)
                pred_fake = discriminator(fake_AB)
                generator_loss = -torch.mean(pred_fake)
                bce_loss = criterion(masks_pred, masks)
                total_loss = generator_loss + bce_loss * lambda_bce
                
                optimizer_generator.zero_grad()
                total_loss.backward()
                optimizer_generator.step()

            if idx % 100 == 0:
                print('train step[%d/%d] discriminator loss: %.5f, total loss: %.5f, generator loss: %.5f, bce loss: %.5f, time: %.2f' % 
                            (idx, len(train_loader), discriminator_loss.item(), total_loss.item(), generator_loss.item(), bce_loss.item(), time.time() - epoch_start_time))

            if epoch % 10 == 0 and idx % 100 == 0:
                rand_idx_start = randrange(masks.size(0) - 3)
                rand_idx_end = rand_idx_start + 3
                test_masks_pred = torch.sigmoid(masks_pred[rand_idx_start:rand_idx_end]).detach().cpu()
                test_masks_pred = test_masks_pred.permute(0, 2, 3, 1).numpy().astype(np.float32)
                test_masks_pred = np.squeeze(test_masks_pred, axis=-1)

                test_masks = masks[rand_idx_start:rand_idx_end].permute(0, 2, 3, 1).cpu().numpy().astype(np.float32)
                test_masks = np.squeeze(test_masks, axis=-1)

                test_images = images[rand_idx_start:rand_idx_end].permute(0, 2, 3, 1).cpu().numpy()
                test_images = test_images * imagenet_std + imagenet_mean
                test_images = np.maximum(test_images, 0.0)
                test_images = np.minimum(test_images, 1.0)
                sample_images(epoch, idx, test_images, test_masks, test_masks_pred, image_save_path)
            # break
        
        # eval
        model.eval()
        total_fmeasure = 0.0
        total_image_number = 0
        random_number = randrange(len(image_test_list))
        for eval_idx, (image_test, mask_test) in enumerate(image_test_list):
            image = cv2.imread(image_test)
            h, w, _ = image.shape
            min_length = min(h, w)
            max_length = max(h, w)

            # pass global prediction
            if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
                continue

            image_name = image_test.split('/')[-1].split('.')[0]

            gt_mask = cv2.imread(mask_test, cv2.IMREAD_GRAYSCALE)

            if min_length * padding_resize_ratio < max_length:
                image, _ = image_padding(image)
                gt_mask, _ = image_padding(gt_mask, is_mask=True)

            image = cv2.resize(image, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.resize(gt_mask, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            gt_mask = cv2.erode(gt_mask, kernel, iterations=1)

            image = preprocess_input(image, input_space="BGR")
            image = np.expand_dims(image, axis=0)
            with torch.no_grad():
                image = torch.from_numpy(image).permute(0, 3, 1, 2).float().to(device)
                pred = torch.sigmoid(model(image)).cpu()

            out_img = pred[0].permute(1, 2, 0).numpy() * 255
            out_img = out_img.astype(np.uint8)
            out_img[out_img > value] = 255
            out_img[out_img <= value] = 0

            # if random_number == 0:
            #     cv2.imwrite('%s/%d_%s.png' % (image_save_path, epoch, image_name), out_img)

            gt_mask = np.expand_dims(gt_mask, axis=-1)
            # f_measure
            # background 1, text 0
            gt_mask[gt_mask > 0] = 1
            out_img[out_img > 0] = 1

            # true positive
            tp = np.zeros(gt_mask.shape, np.uint8)
            tp[(out_img==0) & (gt_mask==0)] = 1
            numtp = tp.sum()

            # false positive
            fp = np.zeros(gt_mask.shape, np.uint8)
            fp[(out_img==0) & (gt_mask==1)] = 1
            numfp = fp.sum()

            # false negative
            fn = np.zeros(gt_mask.shape, np.uint8)
            fn[(out_img==1) & (gt_mask==0)] = 1
            numfn = fn.sum()

            precision = numtp / float(numtp + numfp)
            recall = numtp / float(numtp + numfn)
            fmeasure = 100. * (2. * recall * precision) / (recall + precision) # percent

            total_fmeasure += fmeasure
            total_image_number += 1
            # break
        total_fmeasure /= total_image_number

        if best_fmeasure < total_fmeasure:
            best_fmeasure = total_fmeasure

        print('epoch[%d/%d] fmeasure: %.4f, best_fmeasure: %.4f, time: %.2f' 
                    % (epoch + 1, epochs, total_fmeasure, best_fmeasure, time.time() - epoch_start_time))
        print()

    torch.save(model.state_dict(), weight_path + '/unet_global_%d_%.4f.pth' % (epoch + 1, total_fmeasure))
    torch.save(discriminator.state_dict(), weight_path + '/dis_global_%d_%.4f.pth' % (epoch + 1, total_fmeasure))

if __name__ == "__main__":
    base_model_list = ['efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 'efficientnet-b3'
                'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'resnet50', 'timm-efficientnet-b0',
                'se_resnext101_32x4d']

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='3', help="GPU number")
    parser.add_argument("--epochs", type=int, default=150, help="epochs")
    parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
    parser.add_argument('--base_model_name', type=str, default='efficientnet-b4', help='base_model_name')
    parser.add_argument('--encoder_weights', type=str, default='imagenet', help='none or imagenet')
    parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='discriminator learning rate')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    
    # data set
    parser.add_argument('--image_train_dir', type=str, default='/data/denoise/Label_resize/image', help='512 resized image train dir')
    parser.add_argument('--mask_train_dir', type=str, default='/data/denoise/Label_resize/mask', help='512 resized mask train dir')
    parser.add_argument('--original_dir', type=str, default='/mnt/nas/data/denoise/Label_data', help='original dir - subdir should has image, mask')

    parser.add_argument("--fold_num", type=int, default=0, help="fold number, 0 ~ fold_total-1")
    parser.add_argument("--fold_total", type=int, default=5, help="fold total")

    opt = parser.parse_args()
    unet_train(opt.epochs, opt.gpu, opt.base_model_name, opt.encoder_weights, opt.generator_lr, opt.discriminator_lr, opt.lambda_bce,
                opt.batch_size, opt.image_train_dir, opt.mask_train_dir, opt.original_dir, opt.fold_num, opt.fold_total)