import os
import numpy as np
import cv2
from tool_clean import check_is_image, image_padding


# resize label dataset 512
def main(argv=None):

    image_dir = '/mnt/nas/data/denoise/Label_data/image'
    mask_dir = '/mnt/nas/data/denoise/Label_data/mask'

    image_pathes = os.listdir(image_dir)
    image_list = [] 
    for image_path in image_pathes:
        if not check_is_image(image_path):
            print('not image', image_path)
            continue
        image_list.append( (os.path.join(image_dir, image_path), os.path.join(mask_dir, image_path)) )

    imgh = 512
    imgw = 512
    reshape = (imgh, imgw)
    rotation = [0, 1, 2, 3]

    skip_resize_ratio = 6
    skip_max_length = 512
    padding_resize_ratio = 4
    kernel = np.ones((5, 5), np.uint8)
    
    # 1,818
    image_save_dir = '/data/denoise/Label_resize/image'
    mask_save_dir = '/data/denoise/Label_resize/mask'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    for image, mask in image_list:
        img_name = image.split('/')[-1].split('.')[0]

        image = cv2.imread(image)
        h, w = image.shape[:2]
        min_length = min(h, w)
        max_length = max(h, w)

        # pass global prediction
        if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
            continue

        mask = cv2.imread(mask, cv2.IMREAD_GRAYSCALE)
        mask[mask < 128] = 0
        mask[mask >= 128] = 255

        if min_length * padding_resize_ratio < max_length:
            mask, _ = image_padding(mask, is_mask=True)
            image, _ = image_padding(image)

        print('processing the image:', img_name)
        
        image = cv2.resize(image, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        mask = cv2.erode(mask, kernel, iterations=1)

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

        # exit(1)

if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
