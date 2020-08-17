import numpy as np
import cv2

# utils for binarization
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.bmp', '.BMP'
]


def check_is_image(file_name):
    return any(file_name.endswith(extension) for extension in IMG_EXTENSIONS)


def image_padding(image, is_mask=False, is_gray=False):

    if is_mask:
        median = 255 # white background
    else:
        if is_gray:
            median = getMedian(image, 0)
        else:
            median = (getMedian(image, 0), getMedian(image, 1), getMedian(image, 2))

    input_img_height, input_img_width = image.shape[:2]

    if input_img_width >= input_img_height:
        new_size = input_img_width
        x_offset = 0
        y_offset = int((input_img_width-input_img_height)/2)
    elif input_img_height>input_img_width:
        new_size = input_img_height
        x_offset = int((input_img_height-input_img_width)/2)
        y_offset = 0
    
    if is_mask or is_gray:
        new_image = np.ones((new_size,new_size), np.uint8) * median
        new_image[y_offset:y_offset+input_img_height, x_offset:x_offset+input_img_width]=image
    else:
        new_image_b = np.ones((new_size, new_size, 1), np.uint8) * median[0]
        new_image_g = np.ones((new_size, new_size, 1), np.uint8) * median[1]
        new_image_r = np.ones((new_size, new_size, 1), np.uint8) * median[2]
        new_image   = np.concatenate([new_image_b, new_image_g, new_image_r], axis=2)
        new_image[y_offset:y_offset+input_img_height, x_offset:x_offset+input_img_width, :]=image
    
    return new_image, (y_offset, y_offset+input_img_height, x_offset, x_offset+input_img_width)


def getMedian(img, ch):
    h, w = img.shape[0:2]
    m = w * h / 2
    bin = 0
    med = -1.0

    size = 256
    hist = cv2.calcHist([img], [ch], None, [size], [0, 256])

    i = 0
    while i < size:
        bin += np.round(hist[i])
        if bin > m:
            med = i
            break
        i += 1

    return med


# get patch with padding
def get_image_patch(image, target_h, target_w, overlap, is_mask, is_gray=False):
    image_h, image_w = image.shape[:2]
    overlap_h = int(target_h * overlap)
    overlap_w = int(target_w * overlap)

    if is_mask:
        median = 255 # white background
    else:
        if is_gray:
            median = getMedian(image, 0)
        else:
            median = (getMedian(image, 0), getMedian(image, 1), getMedian(image, 2))
    
    image_list = []
    posit_list = []

    for start_h in range(0, image_h - target_h, overlap_h):
        end_h = start_h + target_h
        if end_h > image_h:
            end_h = image_h
        for start_w in range(0, image_w - target_w, overlap_w):
            end_w = start_w + target_w
            if end_w > image_w:
                end_w = image_w

            imgpath = image[start_h:end_h, start_w:end_w]
            image_list.append(imgpath)
            pos = np.array([start_h, start_w, end_h, end_w, 0, 0])
            posit_list.append(pos)

    # last coloum 
    for start_w in range(0, image_w - target_w, overlap_w):
        end_w = start_w + target_w
        if end_w > image_w:
            end_w = image_w

        end_h = image_h 
        start_h = end_h - target_h
        if start_h < 0:
            start_h = 0

        imgpath = image[start_h:end_h, start_w:end_w]

        # centered image, when image size is smaller than target_h, w
        w_len = (end_w - start_w)
        h_len = (end_h - start_h)
        if w_len < target_w or h_len < target_h:
            # filled with median value
            if is_mask or is_gray:
                patch = np.ones((target_h, target_w), np.uint8) * median
            else:
                patch_b = np.ones((target_h, target_w, 1), np.uint8) * median[0]
                patch_g = np.ones((target_h, target_w, 1), np.uint8) * median[1]
                patch_r = np.ones((target_h, target_w, 1), np.uint8) * median[2]
                patch   = np.concatenate([patch_b, patch_g, patch_r], axis=2)

            w_shift = ( target_w - w_len ) // 2
            h_shift = ( target_h - h_len ) // 2
            patch[h_shift:h_shift + h_len, w_shift:w_shift + w_len] = imgpath
            image_list.append(patch)
            pos = np.array([start_h, start_w, end_h, end_w, h_shift, w_shift])
            posit_list.append(pos)
        else:
            image_list.append(imgpath)
            pos = np.array([start_h, start_w, end_h, end_w, 0, 0])
            posit_list.append(pos)
        
    # last row 
    for start_h in range(0, image_h - target_h, overlap_h):
        end_h = start_h + target_h
        if end_h > image_h:
            end_h = image_h

        end_w = image_w
        start_w = end_w - target_w
        if start_w < 0:
            start_w = 0

        imgpath = image[start_h:end_h, start_w:end_w]

        w_len = (end_w - start_w)
        h_len = (end_h - start_h)
        if w_len < target_w or h_len < target_h:
            # filled with median value
            if is_mask or is_gray:
                patch = np.ones((target_h, target_w), np.uint8) * median
            else:
                patch_b = np.ones((target_h, target_w, 1), np.uint8) * median[0]
                patch_g = np.ones((target_h, target_w, 1), np.uint8) * median[1]
                patch_r = np.ones((target_h, target_w, 1), np.uint8) * median[2]
                patch   = np.concatenate([patch_b, patch_g, patch_r], axis=2)

            w_shift = ( target_w - w_len ) // 2
            h_shift = ( target_h - h_len ) // 2
            patch[h_shift:h_shift + h_len, w_shift:w_shift + w_len] = imgpath
            image_list.append(patch)
            pos = np.array([start_h, start_w, end_h, end_w, h_shift, w_shift])
            posit_list.append(pos)
        else:
            image_list.append(imgpath)
            pos = np.array([start_h, start_w, end_h, end_w, 0, 0])
            posit_list.append(pos)

    # last rectangle
    end_h = image_h
    start_h = end_h - target_h
    if start_h < 0:
        start_h = 0

    end_w = image_w 
    start_w = end_w - target_w
    if start_w < 0:
        start_w = 0
        
    imgpath = image[start_h:end_h, start_w:end_w]

    w_len = (end_w - start_w)
    h_len = (end_h - start_h)
    if w_len < target_w or h_len < target_h:
        # filled with median value
        if is_mask or is_gray:
            patch = np.ones((target_h, target_w), np.uint8) * median
        else:
            patch_b = np.ones((target_h, target_w, 1), np.uint8) * median[0]
            patch_g = np.ones((target_h, target_w, 1), np.uint8) * median[1]
            patch_r = np.ones((target_h, target_w, 1), np.uint8) * median[2]
            patch   = np.concatenate([patch_b, patch_g, patch_r], axis=2)

        w_shift = ( target_w - w_len ) // 2
        h_shift = ( target_h - h_len ) // 2
        patch[h_shift:h_shift + h_len, w_shift:w_shift + w_len] = imgpath
        image_list.append(patch)
        pos = np.array([start_h, start_w, end_h, end_w, h_shift, w_shift])
        posit_list.append(pos)
    else:
        image_list.append(imgpath)
        pos = np.array([start_h, start_w, end_h, end_w, 0, 0])
        posit_list.append(pos)
            
    return image_list, posit_list


# get image patch, no padding
def get_image_patch_deep(image, imgh, imgw, reshape=None, overlap=0.1):

    overlap_wid = int(imgw * overlap)
    overlap_hig = int(imgh * overlap) 

    height, width = image.shape[:2]

    image_list = []
    posit_list = []

    for ys in range(0, height-imgh, overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        for xs in range(0,width-imgw,overlap_wid):
            xe = xs + imgw
            if xe > width:
                xe = width
            imgpath = image[ys:ye,xs:xe]
            if reshape is not None:
                imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
            image_list.append(imgpath)
            pos = np.array([ys,xs,ye,xe])
            posit_list.append(pos)

    # last coloum 
    for xs in range(0, width-imgw, overlap_wid):
        xe = xs + imgw
        if xe > width:
            xe = width
        ye = height 
        ys = ye - imgh
        if ys < 0:
            ys = 0
            
        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)
        
    # last row 
    for ys in range(0, height-imgh, overlap_hig):
        ye = ys + imgh
        if ye > height:
            ye = height
        xe = width
        xs = xe - imgw
        if xs < 0:
            xs = 0
            
        imgpath = image[ys:ye,xs:xe]
        if reshape is not None:
            imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
        image_list.append(imgpath)
        pos = np.array([ys,xs,ye,xe])
        posit_list.append(pos)

    # last rectangle
    ye = height
    ys = ye - imgh
    if ys < 0:
        ys = 0
    xe = width 
    xs = xe - imgw
    if xs < 0:
        xs = 0
        
    imgpath = image[ys:ye,xs:xe]
    if reshape is not None:
        imgpath = cv2.resize(imgpath, dsize=reshape, interpolation=cv2.INTER_NEAREST)
    image_list.append(imgpath)
    pos = np.array([ys,xs,ye,xe])
    posit_list.append(pos)

    return image_list, posit_list