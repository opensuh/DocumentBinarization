import os
from torch.utils.data import Dataset
import numpy as np
import cv2
import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn

import sys
sys.path.append('../Common')
from tool_clean import check_is_image


class Dataset_Return_Four(Dataset):

    def __init__(self, image_dir, mask_dir, original_dir, 
                    base_model_name, encoder_weights, threshold=0.30,
                    is_train=True, fold_num=0, fold_total=5):
        self.threshold = threshold
        self.preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

        self.image_patches = []
        self.mask_patches = []

        original_image_name = []
        origianl_image_pathes = os.listdir(original_dir)
        origianl_image_pathes = sorted(origianl_image_pathes)
        for origianl_image_path in origianl_image_pathes:
            if not check_is_image(origianl_image_path):
                print(origianl_image_path, 'not image')
                continue
            original_image_name.append(origianl_image_path.split('.')[0])

        test_image_name = original_image_name[fold_num::fold_total]
        train_image_name = [i for i in original_image_name if i not in test_image_name]
        print('total image len:', len(original_image_name), 
                'train len: %d' % len(train_image_name) if is_train else 'test len: %d' % len(self.image_patches))

        cnt = 0
        images = os.listdir(image_dir)

        # s[i:j:k] slice of s from i to j with step k
        for image in images:
            if not check_is_image(image):
                print(image, 'not image')
                continue

            if not os.path.isfile(os.path.join(mask_dir, image)):
                print(image, 'no mask')
                continue
            
            if is_train and '_'.join(image.split('_')[:-1]) in train_image_name:
                self.image_patches.append(os.path.join(image_dir, image))
                self.mask_patches.append(os.path.join(mask_dir, image))
            elif not is_train and '_'.join(image.split('_')[:-1]) in test_image_name:
                self.image_patches.append(os.path.join(image_dir, image))
                self.mask_patches.append(os.path.join(mask_dir, image))
            cnt += 1

            if cnt % 10000 == 0:
                print(cnt)
                # break
        print('total patch len:', cnt, 'train patch len:' if is_train else 'test patch len:', len(self.image_patches))

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, i):
        image = cv2.imread(self.image_patches[i])
        mask = cv2.imread(self.mask_patches[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_gray = np.expand_dims(image_gray, axis=-1)
        image_blue = image[:, :, 0:1]
        image_green = image[:, :, 1:2]
        image_red = image[:, :, 2:3]

        mask_and_blue = np.bitwise_and(image_blue, mask)
        val = np.max(mask_and_blue) * self.threshold
        mask_and_blue[mask_and_blue <= val] = 0
        mask_and_blue[mask_and_blue > val] = 255

        mask_and_green = np.bitwise_and(image_green, mask)
        val = np.max(mask_and_green) * self.threshold
        mask_and_green[mask_and_green <= val] = 0
        mask_and_green[mask_and_green > val] = 255

        mask_and_red = np.bitwise_and(image_red, mask)
        val = np.max(mask_and_red) * self.threshold
        mask_and_red[mask_and_red <= val] = 0
        mask_and_red[mask_and_red > val] = 255
        
        image_blue = self.preprocess_input(image_blue)
        image_green = self.preprocess_input(image_green)
        image_red = self.preprocess_input(image_red)
        image_gray = self.preprocess_input(image_gray)

        image_blue = torch.from_numpy(image_blue).permute(2, 0, 1).float()
        mask_and_blue = torch.from_numpy(mask_and_blue).permute(2, 0, 1).float() / 255.

        image_green = torch.from_numpy(image_green).permute(2, 0, 1).float()
        mask_and_green = torch.from_numpy(mask_and_green).permute(2, 0, 1).float() / 255.
        
        image_red = torch.from_numpy(image_red).permute(2, 0, 1).float()
        mask_and_red = torch.from_numpy(mask_and_red).permute(2, 0, 1).float() / 255.

        image_gray = torch.from_numpy(image_gray).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.

        return (image_blue, image_green, image_red, image_gray), (mask_and_blue, mask_and_green, mask_and_red, mask)


class Dataset_Return_One(Dataset):

    def __init__(self, image_dir, mask_dir, original_dir, 
                    base_model_name, encoder_weights,
                    is_train=True, fold_num=0, fold_total=5):
        self.base_model_name = base_model_name
        self.preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

        self.image_patches = []
        self.mask_patches = []

        original_image_name = []
        origianl_image_pathes = os.listdir(original_dir)
        origianl_image_pathes = sorted(origianl_image_pathes)
        for origianl_image_path in origianl_image_pathes:
            if not check_is_image(origianl_image_path):
                print(origianl_image_path, 'not image')
                continue
            original_image_name.append(origianl_image_path.split('.')[0])

        test_image_name = original_image_name[fold_num::fold_total]
        train_image_name = [i for i in original_image_name if i not in test_image_name]
        print('total image len:', len(original_image_name), 
                'train len: %d' % len(train_image_name) if is_train else 'test len: %d' % len(self.image_patches))

        cnt = 0
        images = os.listdir(image_dir)

        # s[i:j:k] slice of s from i to j with step k
        for image in images:
            if not check_is_image(image):
                print(image, 'not image')
                continue

            if not os.path.isfile(os.path.join(mask_dir, image)):
                print(image, 'no mask')
                continue
            
            if is_train and '_'.join(image.split('_')[:-1]) in train_image_name:
                self.image_patches.append(os.path.join(image_dir, image))
                self.mask_patches.append(os.path.join(mask_dir, image))
            elif not is_train and '_'.join(image.split('_')[:-1]) in test_image_name:
                self.image_patches.append(os.path.join(image_dir, image))
                self.mask_patches.append(os.path.join(mask_dir, image))
            cnt += 1

            if cnt % 10000 == 0:
                print(cnt)
                # break
        print('total patch len:', cnt, 'train patch len:' if is_train else 'test patch len:', len(self.image_patches))

    def __len__(self):
        return len(self.image_patches)

    def __getitem__(self, i):
        image = cv2.imread(self.image_patches[i])
        mask = cv2.imread(self.mask_patches[i], cv2.IMREAD_GRAYSCALE)
        mask = np.expand_dims(mask, axis=-1)

        image = self.preprocess_input(image, input_space="BGR")
        image = torch.from_numpy(image).permute(2, 0, 1).float()

        mask = torch.from_numpy(mask).permute(2, 0, 1).float() / 255.

        return image, mask