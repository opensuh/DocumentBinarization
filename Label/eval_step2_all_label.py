import os
import numpy as np
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import time
import cv2
import csv
import argparse
import pytesseract

import sys
sys.path.append('../Common')
from tool_clean import get_image_patch, check_is_image, image_padding
from metrics import get_metric, get_tesseract_data_name, get_levenshtein


parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='0', help="GPU number")
parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
parser.add_argument('--base_model_name', type=str, default='efficientnet-b4', help='base_model_name')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='encoder_weights')
parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='discriminator learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--threshold', type=float, default=0.30, help='threshold for bgr mask')

parser.add_argument("--fold_num", type=int, default=0, help="fold number")
parser.add_argument("--fold_total", type=int, default=5, help="fold total")

# data set
parser.add_argument('--original_dir', type=str, default='/data/denoise/Label_data/', help='original dir - subdir must has image, mask')
# tesseract option and train_data folder
custom_oem_psm_config = r'--oem 1 --psm 6 --tessdata-dir "/mnt/nas/data/denoise/pytesseract_data"'

opt = parser.parse_args()

device = torch.device("cuda:%s" % opt.gpu)
base_model_name = opt.base_model_name
lambda_bce = opt.lambda_bce
generator_lr = opt.generator_lr
threshold = opt.threshold
encoder_weights = opt.encoder_weights
fold_num = opt.fold_num
fold_total = opt.fold_total
preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

# load step1
weight_folder = ('./step1_label%d_' % fold_num) + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr) + '_' + str(threshold)
weight_list = os.listdir(weight_folder)
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list if 'unet' in weight_path]
weight_list = sorted(weight_list)
print('step1 weight', weight_list)

models = []

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

# step2 unet
weight_folder = ('./step2_label%d_' % fold_num) + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr)
weight_list = os.listdir(weight_folder)
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list if 'unet' in weight_path]
weight_list = sorted(weight_list)
print('step2 weight', weight_list)

model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()

# step2 global model
weight_folder = ('./step2_resize_label_%d_' % fold_num) + base_model_name + '_' + str(int(lambda_bce)) + '_' + str(generator_lr)
weight_list = os.listdir(weight_folder)
weight_list = [os.path.join(weight_folder, weight_path) for weight_path in weight_list if 'unet' in weight_path]
weight_list = sorted(weight_list)
print('step2 global weight', weight_list)

model_normal_resize = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model_normal_resize.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model_normal_resize.to(device)
model_normal_resize.requires_grad_(False)
model_normal_resize.eval()

# get test image
original_dir = opt.original_dir
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

# test parameter
value = int(256 * 0.5)
batch_size = 16
kernel = np.ones((7, 7), np.uint8)
resize_size = (512, 512)
skip_resize_ratio = 6
skip_max_length = 512
padding_resize_ratio = 4

# make directoies
save_root_dir = './predicted_image_label_%d' % fold_num
os.makedirs(save_root_dir, exist_ok=True)

# save fmeasure
save_fmeasure = {
    'step2_normal': [[] for i in range(4)]
}

save_csv = open('./%s/metrics.csv' % save_root_dir, 'w')
save_csv_file = csv.writer(save_csv)
save_csv_file.writerow([' ', 'step2_normal', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])
save_average_time = 0.0
total_image_num = 0
# end fmeasure

save_step2_or_normal_dir = os.path.join(save_root_dir, 'step2_normal')
os.makedirs(save_step2_or_normal_dir, exist_ok=True)
# end directories

# for ocr
ocr_gt = {}
csv_file = open('/mnt/nas/data/denoise/label_ocr_gt.csv', encoding='utf-8-sig')
csv_reader = csv.reader(csv_file, delimiter=',')
for row in csv_reader:
    ocr_gt[''.join(row[0].split())] = ''.join(row[3].split())

save_ocr_csv = open('./%s/metrics_ocr.csv' % save_root_dir, 'w')
save_ocr_csv_file = csv.writer(save_ocr_csv)
save_ocr_csv_file.writerow([' ', 'kor', 'latin', 'total'])
kor_ocr_sum = 0.0
kor_total_num = 0
latin_ocr_sum = 0.0
latin_total_num = 0
total_ocr_sum = 0.0
total_total_num = 0

for test_image, test_mask in image_test_list:
    img_name = test_image.split('/')[-1].split('.')[0]

    image = cv2.imread(test_image)
    h, w = image.shape[:2]

    gt_mask = cv2.imread(test_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask[gt_mask > 0] = 1

    print('processing the image:', img_name)

    start_time = time.time()
    # start step1
    image_patches, poslist = get_image_patch(image, 256, 256, overlap=0.5, is_mask=False)
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

                # from NHWC to NCHW
                target = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float()
                preds.extend( torch.sigmoid(models[channel](target.to(device))).cpu() )
                step += batch_size

        # handling overlap
        out_img = np.ones((h, w, 1)) * 255
        for i in range(len(image_patches)):
            patch = preds[i].permute(1, 2, 0).numpy() * 255

            start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
            h_cut = end_h - start_h
            w_cut = end_w - start_w

            out_img[start_h:end_h, start_w:end_w] = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])

        # for step2
        out_imgs.append(out_img)

    # step1 merged color image
    merge_img[:, :, 0:1] = (out_imgs[0] + out_imgs[3]) / 2.
    merge_img[:, :, 1:2] = (out_imgs[1] + out_imgs[3]) / 2.
    merge_img[:, :, 2:3] = (out_imgs[2] + out_imgs[3]) / 2.
    merge_img = merge_img.astype(np.uint8)
    # end step1
    
    # step2 start
    image_patches, poslist = get_image_patch(merge_img, 256, 256, overlap=0.5, is_mask=False)

    color_patches = []
    for patch in image_patches:
        color_patches.append(preprocess_input(patch.astype(np.float32), input_space="BGR"))

    step = 0
    preds = []
    with torch.no_grad():
        while step < len(image_patches):
            ps = step
            pe = step + batch_size
            if pe >= len(image_patches):
                pe = len(image_patches)

            image_gray = torch.from_numpy(np.array(color_patches[ps:pe])).permute(0, 3, 1, 2).float().to(device)
            preds.extend( torch.sigmoid(model(image_gray)).cpu() )
            step += batch_size

    # handling overlap
    step2_out_img = np.ones((h, w, 1)) * 255
    for i in range(len(image_patches)):
        patch = preds[i].permute(1, 2, 0).numpy() * 255

        start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
        h_cut = end_h - start_h
        w_cut = end_w - start_w

        tmp = np.minimum(out_img[start_h:end_h, start_w:end_w], patch[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut])
        step2_out_img[start_h:end_h, start_w:end_w] = tmp

    step2_out_img = out_img.astype(np.uint8)
    step2_out_img[step2_out_img > value] = 255
    step2_out_img[step2_out_img <= value] = 0
    step2_out_img = np.squeeze(step2_out_img, axis=-1)
    # end step2
    
    # start step2 global
    min_length = min(h, w)
    max_length = max(h, w)
    if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
        pass
    else:
        is_padded = False
        if min_length * padding_resize_ratio < max_length:
            image, position = image_padding(image)
            padded_size = image.shape[0]
            is_padded = True

        resized_img = cv2.resize(image, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
        resized_img = preprocess_input(resized_img, input_space="BGR")
        resized_img = np.expand_dims(resized_img, axis=0)
        resized_img = torch.from_numpy(resized_img).permute(0, 3, 1, 2).float().to(device)
        with torch.no_grad():
            resized_mask_pred = model_normal_resize(resized_img)
            resized_mask_pred = torch.sigmoid(resized_mask_pred).cpu()
        
        resized_mask_pred = resized_mask_pred[0].permute(1, 2, 0).numpy() * 255
        resized_mask_pred = resized_mask_pred.astype(np.uint8)
        resized_mask_pred[resized_mask_pred > value] = 255
        resized_mask_pred[resized_mask_pred <= value] = 0

        if is_padded:
            # restore image padding, size
            resized_mask_pred = cv2.resize(resized_mask_pred, dsize=(padded_size, padded_size), interpolation=cv2.INTER_NEAREST)
            resized_mask_pred = resized_mask_pred[position[0]:position[1], position[2]:position[3]]
        else :
            resized_mask_pred = cv2.resize(resized_mask_pred, dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        
        resized_mask_pred = cv2.erode(resized_mask_pred, kernel, iterations=1)

    save_average_time += (time.time() - start_time)
    total_image_num += 1

    if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
        # allocate new memory
        step2_gray_img = np.copy(step2_out_img)
        step2_gray_img[step2_gray_img > 0] = 1
        step2_fmeasure, step2_pfmeasure, step2_psnr, step2_drd = get_metric(step2_gray_img, gt_mask)

        cv2.imwrite('%s/step2_normal/%s.png' % (save_root_dir, img_name), step2_out_img)
        save_fmeasure['step2_normal'][0].append(step2_fmeasure)
        save_fmeasure['step2_normal'][1].append(step2_pfmeasure)
        save_fmeasure['step2_normal'][2].append(step2_psnr)
        save_fmeasure['step2_normal'][3].append(step2_drd)
        csv_tmp = [img_name, step2_fmeasure, step2_pfmeasure, step2_psnr, step2_drd]
    else:
        step2_normal_or_img = np.bitwise_or(resized_mask_pred, step2_out_img)

        step2_normal_or_img_metric = np.copy(step2_normal_or_img)
        step2_normal_or_img_metric[step2_normal_or_img_metric > 0] = 1

        step2_normal_fmeasure, step2_normal_pfmeasure, step2_normal_psnr, step2_normal_drd = get_metric(step2_normal_or_img_metric, gt_mask)
        cv2.imwrite('%s/step2_normal/%s.png' % (save_root_dir, img_name), step2_normal_or_img)
        save_fmeasure['step2_normal'][0].append(step2_normal_fmeasure)
        save_fmeasure['step2_normal'][1].append(step2_normal_pfmeasure)
        save_fmeasure['step2_normal'][2].append(step2_normal_psnr)
        save_fmeasure['step2_normal'][3].append(step2_normal_drd)
        csv_tmp = [img_name, step2_normal_fmeasure, step2_normal_pfmeasure, step2_normal_psnr, step2_normal_drd]


    contry_name = img_name[11:14]
    if min_length * skip_resize_ratio < max_length or max_length < skip_max_length:
        ocr_temp = pytesseract.image_to_string(step2_out_img, lang=get_tesseract_data_name(contry_name), config=custom_oem_psm_config)
    else:
        ocr_temp = pytesseract.image_to_string(step2_normal_or_img, lang=get_tesseract_data_name(contry_name), config=custom_oem_psm_config)
    ocr_score = get_levenshtein(''.join(ocr_temp.split()), ocr_gt[img_name])

    if contry_name == 'KOR':
        kor_ocr_sum += ocr_score
        kor_total_num += 1
    else:
        latin_ocr_sum += ocr_score
        latin_total_num += 1
    total_ocr_sum += ocr_score
    total_total_num += 1
    # end step2 global
    
    save_csv_file.writerow(csv_tmp)
    # break

csv_tmp = ['average']
for sub_dir in save_fmeasure:
    for sub_list in save_fmeasure[sub_dir]:
        csv_tmp.append( sum(sub_list) / len(sub_list) )
    csv_tmp.extend([' ', ' '])
save_csv_file.writerow(csv_tmp)
save_csv_file.writerow(['average time', save_average_time / total_image_num])
save_csv.close()

if kor_total_num == 0:
    kor_average_ocr = 0.0
else:
    kor_average_ocr = kor_ocr_sum / kor_total_num

if latin_total_num == 0:
    latin_average_ocr = 0.0
else:
    latin_average_ocr = latin_ocr_sum / latin_total_num

total_ocr_sum /= total_total_num

save_ocr_csv_file.writerow([' ', kor_average_ocr, latin_average_ocr, total_ocr_sum])
save_ocr_csv.close()