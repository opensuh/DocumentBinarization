import os
import numpy as np
import cv2
from tool_clean import check_is_image


# LRDE 2514 x 3512 --> resize all
# resize LRDE images 512
def main(argv=None):

    image_dir = '/mnt/nas/data/denoise/LRDE/image/'
    mask_dir = '/mnt/nas/data/denoise/LRDE/mask/'
    imgh = 512
    imgw = 512
    resize_size = (imgh, imgw)
    rotation = [0, 1, 2, 3]
    kernel = np.ones((5, 5), np.uint8)

    image_save_dir = '/data/denoise/LRDE/resize/image'
    mask_save_dir = '/data/denoise/LRDE/resize/mask'
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(mask_save_dir, exist_ok=True)

    image_list = os.listdir(image_dir)
    for image in image_list:
        if not check_is_image(image):
            print('not image', image)
            
        img_name = image.split('.')[0]

        mask = cv2.imread(os.path.join(mask_dir, image), cv2.IMREAD_GRAYSCALE)

        # there are few images that have a value (1 ~ 254)
        mask[mask < 128] = 0
        mask[mask >= 128] = 255

        image = cv2.imread(os.path.join(image_dir, image))

        print('processing the image:', img_name)           
        
        resized_image = cv2.resize(image, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.resize(mask, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
        resized_mask = cv2.erode(resized_mask, kernel, iterations=1)

        for k in rotation:
            img_tmp = np.rot90(resized_image, k)
            mask_tmp = np.rot90(resized_mask, k)
            cv2.imwrite('%s/%s_r%d.png' % (image_save_dir, img_name, k), img_tmp)
            cv2.imwrite('%s/%s_r%d.png' % (mask_save_dir, img_name, k), mask_tmp)
            
        # vertical axis
        img_tmp = np.fliplr(resized_image)
        mask_tmp = np.fliplr(resized_mask)
        cv2.imwrite('%s/%s_v%d.png' % (image_save_dir, img_name, 0), img_tmp)
        cv2.imwrite('%s/%s_v%d.png' % (mask_save_dir, img_name, 0), mask_tmp)
        
        # horizontal axis
        img_tmp = np.flipud(resized_image)
        mask_tmp = np.flipud(resized_mask)
        cv2.imwrite('%s/%s_h%d.png' % (image_save_dir, img_name, 0), img_tmp)
        cv2.imwrite('%s/%s_h%d.png' % (mask_save_dir, img_name, 0), mask_tmp)
        
        #exit(1)   

if __name__ == '__main__':
    print('make?')
    exit(1)
    main()
