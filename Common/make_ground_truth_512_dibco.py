import os
import numpy as np
import cv2
from tool_clean import check_is_image, image_padding
import csv


# resize DIBCO train set image to 512
def main(argv=None):

    image_dirs = ['/mnt/nas/data/denoise/DIBCO/train/image', '/mnt/nas/data/denoise/DIBCO/test/image']
    mask_dirs = ['/mnt/nas/data/denoise/DIBCO/train/mask', '/mnt/nas/data/denoise/DIBCO/test/mask']

    imgh = 512
    imgw = 512
    reshape = (imgh, imgw)
    rotation = [0, 1, 2, 3]

    skip_resize_ratio = 6
    skip_max_length = 512
    padding_resize_ratio = 4

    # 
    image_save_dirs = ['/data/denoise/dibco_resize_6_4/train/image', '/data/denoise/dibco_resize_6_4/test/image']
    mask_save_dirs = ['/data/denoise/dibco_resize_6_4/train/mask', '/data/denoise/dibco_resize_6_4/test/mask']

    kernel = np.ones((5, 5), np.uint8)
    for image_dir, mask_dir, image_save_dir, mask_save_dir in zip(image_dirs, mask_dirs, image_save_dirs, mask_save_dirs):
        
        os.makedirs(image_save_dir, exist_ok=True)
        os.makedirs(mask_save_dir, exist_ok=True)
        image_list = os.listdir(image_dir)
        for image in image_list:
            if not check_is_image(image):
                print('not image', image)
                continue
            
            img_name = image.split('.')[0]
            # find and read mask file with gray
            if os.path.isfile(os.path.join(mask_dir, img_name + '.png')):
                mask = cv2.imread(os.path.join(mask_dir, img_name + '.png'), cv2.IMREAD_GRAYSCALE)
            elif os.path.isfile(os.path.join(mask_dir, img_name + '.bmp')):
                mask = cv2.imread(os.path.join(mask_dir, img_name + '.bmp'), cv2.IMREAD_GRAYSCALE)
            else:
                print('no mask')
                continue

            # there are few images that have a value (1 ~ 254)
            # bickley image has faint text
            mask[mask < 190] = 0
            mask[mask >= 190] = 255

            image = cv2.imread(os.path.join(image_dir, image))
            h, w = image.shape[:2]
            min_length = min(h, w)
            max_length = max(h, w)

            # pass global prediction
            if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
                continue

            if min_length * padding_resize_ratio < max_length:
                mask, _ = image_padding(mask, is_mask=True)
                image, _ = image_padding(image)

            print('processing the image:', img_name)
            
            image = cv2.resize(image, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(mask, kernel, iterations=1)

            if 'test' in image_dir:
                cv2.imwrite('%s/%s.png' % (image_save_dir, img_name), image)
                cv2.imwrite('%s/%s.png' % (mask_save_dir, img_name), mask)
                continue

            for k in rotation:
                img_tmp = np.rot90(image, k)
                mask_tmp = np.rot90(mask, k)
                cv2.imwrite('%s/%s_r%d.png' % (image_save_dir, img_name, k), img_tmp)
                cv2.imwrite('%s/%s_r%d.png' % (mask_save_dir, img_name, k), mask_tmp)

            # vertical axis
            img_tmp = np.fliplr(image)
            mask_tmp = np.fliplr(mask)
            cv2.imwrite('%s/%s_v%d.png' % (image_save_dir, img_name, 0), img_tmp)
            cv2.imwrite('%s/%s_v%d.png' % (mask_save_dir, img_name, 0), mask_tmp)

            # horizontal axis
            img_tmp = np.flipud(image)
            mask_tmp = np.flipud(mask)
            cv2.imwrite('%s/%s_h%d.png' % (image_save_dir, img_name, 0), img_tmp)
            cv2.imwrite('%s/%s_h%d.png' % (mask_save_dir, img_name, 0), mask_tmp)

            #exit(1)

if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
