import os
import numpy as np
import cv2
import math
from tool_clean import get_image_patch, check_is_image


def main(argv=None):

    image_dir = '/mnt/nas/data/denoise/LRDE/image/'
    mask_dir = '/mnt/nas/data/denoise/LRDE/mask/'

    overlap = 30. / 100. # 30. / 100. -> 317,750
    imgh = 256
    imgw = 256

    image_save_dir = '/data/denoise/LRDE/image_patches'
    mask_save_dir = '/data/denoise/LRDE/mask_patches'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_pathes = os.listdir(image_dir)
    for image_path in image_pathes:
        if not check_is_image(image_path):
            print('not image', image_path)
            continue

        image_name = image_path.split('.')[0]

        mask = cv2.imread(mask_dir + image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.imread(image_dir + image_path)
        
        print('processing the image:', image_path)

        image_patches, _ = get_image_patch(image, imgh, imgw, overlap=overlap, is_mask=False)
        mask_patches, poslist = get_image_patch(mask, imgh, imgw, overlap=overlap, is_mask=True)

        print('get patches: %d' % len(image_patches))
        for idx in range(len(image_patches)):
            img_color = image_patches[idx]
            img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

            mask_gray = mask_patches[idx]

            img_color_tmp = img_color
            mask_gray_tmp = mask_gray
            cv2.imwrite('%s/%s_i%dh0.png' % (image_save_dir, image_name, idx), img_color_tmp)
            cv2.imwrite('%s/%s_i%dh0.png' % (mask_save_dir, image_name, idx), mask_gray_tmp)

            # horizontal axis
            img_color_tmp = np.flipud(img_color)
            mask_gray_tmp = np.flipud(mask_gray)
            cv2.imwrite('%s/%s_i%dh1.png' % (image_save_dir, image_name, idx), img_color_tmp)
            cv2.imwrite('%s/%s_i%dh1.png' % (mask_save_dir, image_name, idx), mask_gray_tmp)

        # exit(1)

if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
