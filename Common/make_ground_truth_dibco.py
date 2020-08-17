import os
import numpy as np
import cv2
import math
from tool_clean import get_image_patch_deep, check_is_image


def main(argv=None):

    image_dir = '/mnt/nas/data/denoise/DIBCO/train/image/'
    mask_dir = '/mnt/nas/data/denoise/DIBCO/train/mask/'

    overlap = 30. / 100. # 30. / 100. -> 119,208
    imgh = 256
    imgw = 256
    reshape = (imgw, imgh)
    scale_list = [0.75, 1.00, 1.25, 1.50] # sample patches with the scale factor and resize patches to 256 * 256 // 192, 256, 320, 384
    rotation = [0, 3]

    image_save_dir = '/data/denoise/DIBCO/train/image_patches'
    mask_save_dir = '/data/denoise/DIBCO/train/mask_patches'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_pathes = os.listdir(image_dir)
    for image_path in image_pathes:
        if not check_is_image(image_path):
            print('not image', image_path)
            continue

        image_name = image_path.split('.')[0]

        # find and read mask file
        if os.path.isfile(os.path.join(mask_dir, image_name) + '.png'):
            mask = cv2.imread(os.path.join(mask_dir, image_name) + '.png', cv2.IMREAD_GRAYSCALE)
        elif os.path.isfile(os.path.join(mask_dir, image_name) + '.bmp'):
            mask = cv2.imread(os.path.join(mask_dir, image_name) + '.bmp', cv2.IMREAD_GRAYSCALE)
        else:
            print('no mask', image_path)
            continue

        # there are few images that have a value (1 ~ 254), bickley image has thin stroke
        mask[mask < 190] = 0
        mask[mask >= 190] = 255

        image = cv2.imread(os.path.join(image_dir + image_path))
        print('processing the image:', image_path)
        # continue

        scale_cnt = 0
        for scale in scale_list:
            # (patches, 256, 256, 3)
            crpW = int(scale * imgw)
            crpH = int(scale * imgh)

            image_patches, _ = get_image_patch_deep(image, crpH, crpW, reshape, overlap=overlap)
            mask_patches, poslist = get_image_patch_deep(mask, crpH, crpW, reshape, overlap=overlap)

            print('get patches: %d' % len(image_patches))
            
            for idx in range(len(image_patches)):
                img_color = image_patches[idx]
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                mask_gray = mask_patches[idx]

                # agumentation
                for k in rotation:
                    img_color_tmp = np.rot90(img_color, k)
                    mask_gray_tmp = np.rot90(mask_gray, k)
                    cv2.imwrite('%s/%s_s%dr%di%d.png' % (image_save_dir, image_name, scale_cnt, k, idx), img_color_tmp)
                    cv2.imwrite('%s/%s_s%dr%di%d.png' % (mask_save_dir, image_name, scale_cnt, k, idx), mask_gray_tmp)

                # exit(1)
            scale_cnt += 1
            # exit(1)


if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
