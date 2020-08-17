import os
import numpy as np
import cv2
import math

from tool_clean import get_image_patch, check_is_image

def main(argv=None):

    image_dir = '/mnt/nas/data/denoise/Label_data/image'
    mask_dir = '/mnt/nas/data/denoise/Label_data/mask'

    overlap = 30. / 100. # 30. / 100. -> 65,247
    imgh = 256
    imgw = 256
    scale_list = [0.75, 1.00, 1.25, 1.50] # sample patches with the scale factor and resize patches to 256 * 256 // 192, 256, 384
    resize_size = (imgh, imgw)

    image_save_dir = '/data/denoise/Label_patch/image_patches'
    mask_save_dir = '/data/denoise/Label_patch/mask_patches'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_pathes = os.listdir(image_dir)
    for image_path in image_pathes:
        if not check_is_image(image_path):
            print('not image', image_path)
            continue

        image_name = image_path.split('.')[0]
        image = cv2.imread(os.path.join(image_dir, image_path))

        # find and read mask file
        if not os.path.isfile(os.path.join(mask_dir, image_path)):
            print(image_path, 'no mask')
            exit(1)

        mask = cv2.imread(os.path.join(mask_dir, image_path), cv2.IMREAD_GRAYSCALE)
        mask[mask <= 128] = 0
        mask[mask > 128] = 255

        if image.shape[:2] != mask.shape[:2]:
            print(image_path, 'size mismatch')
            exit(1)
        
        print('processing the image:', image_path)

        scale_cnt = 0
        for scale in scale_list:
            # (patches, 256, 256, 3)
            crpW = int(scale * imgw)
            crpH = int(scale * imgh)

            image_patches, poslist = get_image_patch(image, crpH, crpW, overlap, False)
            mask_patches, poslist = get_image_patch(mask, crpH, crpW, overlap, True)
            print('get patches: %d' % len(image_patches))
            
            for idx in range(len(image_patches)):
                img_color = image_patches[idx]
                img_color = cv2.resize(img_color, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
                img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

                mask_gray = mask_patches[idx]
                mask_gray = cv2.resize(mask_gray, dsize=resize_size, interpolation=cv2.INTER_NEAREST)

                cv2.imwrite('%s/%s_s%di%d.png' % (image_save_dir, image_name, scale_cnt, idx), img_color)
                cv2.imwrite('%s/%s_s%di%d.png' % (mask_save_dir, image_name, scale_cnt, idx), mask_gray)
                
            scale_cnt += 1
        # exit(1)


if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
