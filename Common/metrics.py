import math
import numpy as np

from bwmorph_thin import bwmorph_thin as bwmorph


def my_xor_infile(u_infile, u0_GT_infile):
    temp_fp_infile = np.zeros(u_infile.shape, np.uint8)
    temp_fp_infile[(u_infile == 0) & (u0_GT_infile == 1)] = 1

    temp_fn_infile = np.zeros(u_infile.shape, np.uint8)
    temp_fn_infile[(u_infile == 1) & (u0_GT_infile == 0)] = 1

    temp_xor_infile = (temp_fp_infile | temp_fn_infile)
    return temp_xor_infile


def get_drd(im, im_gt):
    xm, ym = im.shape
    
    # get NUBN
    blkSize=8 # even number
    # 1, 1 padding 후 중앙
    u0_GT1 = np.zeros((xm + 2, ym + 2), np.uint8)
    u0_GT1[1 : xm + 1, 1 : ym + 1] = im_gt
    NUBN = 0
    blkSizeSQR = blkSize * blkSize
    # matlab 은 배열 인덱스 1 부터 시작, for 문 마지막 포함
    for i in range(1, (xm - blkSize + 2), blkSize):
        for j in range(1, (ym - blkSize + 2), blkSize):
            blkSum = np.sum(u0_GT1[i:i+blkSize, j:j+blkSize])
            if blkSum != 0 and blkSum != blkSizeSQR:
                NUBN += 1

    mask_size = 5 # odd number
    wm = np.zeros((mask_size, mask_size))
    ic = int(mask_size / 2) # center coordinate
    jc = ic
    for i in range(mask_size):
        for j in range(mask_size):
            if i == ic and j == jc:
                continue
            wm[i, j] = 1. / math.sqrt( (i - ic) * (i - ic) + (j - jc) * (j - jc) )
    wm[ic, jc] = 0.
    wnm = wm / np.sum(wm) # % Normalized weight matrix
    
    # 1 ~ xm + 3, 2 ~ xm + 1
    # get sum of DRD_k
    # 2칸씩 padding 후 가운데
    u0_GT_Resized = np.zeros((xm + ic + 2, ym + jc + 2), np.uint8)
    u0_GT_Resized[ic : xm+ic, jc : ym+jc] = im_gt
    u_Resized = np.zeros((xm + ic + 2, ym + jc + 2), np.uint8)
    u_Resized[ic : xm+ic, jc : ym+jc] = im

    temp_fp_Resized = np.zeros(u_Resized.shape, np.uint8)
    temp_fp_Resized[(u_Resized == 0) & (u0_GT_Resized == 1)] = 1
    temp_fn_Resized = np.zeros(u_Resized.shape, np.uint8)
    temp_fn_Resized[(u_Resized == 1) & (u0_GT_Resized == 0)] = 1

    Diff = temp_fp_Resized | temp_fn_Resized
    xm2, ym2 = Diff.shape
    SumDRDk = 0.
    for i in range(ic, xm2 - ic):
        for j in range(jc, ym2 - jc):
            if Diff[i, j] == 1:
                Local_Diff = my_xor_infile(u0_GT_Resized[i - ic : i + ic + 1 , j - ic : j + ic + 1], u_Resized[i:i+1, j:j+1])
                DRDk = np.sum(np.multiply(Local_Diff, wnm))
                SumDRDk += DRDk

    temp_DRD = SumDRDk / NUBN

    return temp_DRD


def get_metric(pred_img, gt_mask):
    # true positive
    tp = np.zeros(gt_mask.shape, np.uint8)
    tp[(pred_img==0) & (gt_mask==0)] = 1
    numtp = tp.sum()

    # false positive
    fp = np.zeros(gt_mask.shape, np.uint8)
    fp[(pred_img==0) & (gt_mask==1)] = 1
    numfp = fp.sum()

    # false negative
    fn = np.zeros(gt_mask.shape, np.uint8)
    fn[(pred_img==1) & (gt_mask==0)] = 1
    numfn = fn.sum()

    precision = numtp / float(numtp + numfp)
    recall = numtp / float(numtp + numfn)
    fmeasure = 100. * (2. * recall * precision) / (recall + precision) # percent

    # get skeletonized im_gt
    sk = bwmorph(1 - gt_mask)
    im_sk = np.ones(gt_mask.shape, np.uint8)
    im_sk[sk] = 0

    # skel true positive
    ptp = np.zeros(gt_mask.shape, np.uint8)
    ptp[(pred_img==0) & (im_sk==0)] = 1
    numptp = ptp.sum()

    # skel false negative
    pfn = np.zeros(gt_mask.shape, np.uint8)
    pfn[(pred_img==1) & (im_sk==0)] = 1
    numpfn = pfn.sum()

    # get pseudo-FMeasure
    precall = numptp / float(numptp + numpfn)
    pfmeasure = 100 * (2 * precall * precision) / (precall + precision) # percent

    # get Peak Signal to Noise Ratio
    # in formula, C -> difference between text and background (1.0 ~ 0.0)
    h, w = gt_mask.shape
    npixel = h * w
    mse = float(numfp + numfn) / npixel
    psnr = 10. * np.log10(1. / mse)

    # get Distance Reciprocal Distortion Metric
    #drd = drd_fn(im, im_gt)
    drd = get_drd(pred_img, gt_mask)

    return fmeasure, pfmeasure, psnr, drd


# https://lovit.github.io/nlp/2018/08/28/levenshtein_hangle/
def cal_levenshtein(s1, s2, debug=False):
    if len(s1) < len(s2):
        return cal_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug:
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]

def get_tesseract_data_name(contry_name):
    if contry_name == 'KOR':
        return 'kor'
    elif contry_name == 'GER':
        return 'deu'
    elif contry_name == 'FRA':
        return 'fra'
    elif contry_name == 'SPA':
        return 'spa'
    elif contry_name == 'USA':
        return 'eng'
    else:
        return 'eng'

def get_levenshtein(s1, s2):
    return (1. - cal_levenshtein(s1, s2) / max(len(s1), len(s2))) * 100


if __name__=="__main__":
    
    get_levenshtein('')