import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import sys
from sklearn.feature_extraction import image
from tqdm import tqdm
from joint_bilateral_filter import Joint_bilateral_filter
import time

def computeDisp(Il, Ir, max_disp,kernel=10):
    JBF = Joint_bilateral_filter(3, 0.1, 'reflect')
    h, w, ch = Il.shape
    left_labels = np.zeros((h, w), dtype=np.float32)
    right_labels = np.zeros((h, w), dtype=np.float32)
    kernel_half = int(kernel / 2)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)
    Il_gray = cv2.cvtColor(Il, cv2.COLOR_BGR2GRAY)
    Ir_gray = cv2.cvtColor(Ir, cv2.COLOR_BGR2GRAY)
    # Il = Il_gray
    # Ir = Ir_gray
    # Il = JBF.joint_bilateral_filter(Il, Il_gray).astype(np.uint8)
    # Ir = JBF.joint_bilateral_filter(Ir, Ir_gray).astype(np.uint8)

    Il = np.pad(Il, ((kernel_half, kernel_half), (kernel_half, kernel_half), (0, 0)), 'symmetric').astype(np.float64)
    Ir = np.pad(Ir, ((kernel_half, kernel_half), (kernel_half, kernel_half), (0, 0)), 'symmetric').astype(np.float64)

    # offset_adjust = 255 / max_disp
    # >>> Cost computation
    # TODO: Compute matching cost from Il and Ir
    # For each pixel in the left image
    # Census Transform
    left_matching_cost = np.zeros((h, w, max_disp), dtype=np.float32)
    right_matching_cost = np.zeros((h, w, max_disp), dtype=np.float32)
    for y in tqdm(range(kernel_half, h + kernel_half)):
        for x in range(kernel_half, w + kernel_half):
            # For each disparity level
            for i_disp, disp in enumerate(range(max_disp)):
                # For each pixel in window
                ####################################################
                # colculate the left disparity
                l_left_center = Il[y, x] # (3) RGB
                l_right_center = Ir[y, x-disp] # (3) RGB
                # left_cencus = np.zeros((kernel*kernel-1, 3), dtype=np.float32)
                # right_cencus = np.zeros((kernel*kernel-1, 3), dtype=np.float32)
                l_combi_cencus = np.ones((kernel*kernel-1, 3), dtype=np.float32)

                # right
                if x+disp < w:
                    r_left_center = Il[y, x+disp] # (3) RGB
                else:
                    r_left_center = Il[y, x+disp-w] # (3) RGB
                r_right_center = Ir[y, x] # (3) RGB
                # left_cencus = np.zeros((kernel*kernel-1, 3), dtype=np.float32)
                # right_cencus = np.zeros((kernel*kernel-1, 3), dtype=np.float32)
                r_combi_cencus = np.ones((kernel*kernel-1, 3), dtype=np.float32)
                # print(y)
                if (x-kernel_half-disp) >= 0:
                    l_tmp_left = Il[y-kernel_half:y+kernel_half+1,x-kernel_half:x+kernel_half+1,:].reshape(-1) > l_left_center.reshape(3,1)
                    l_tmp_right = Ir[y-kernel_half:y+kernel_half+1,x-kernel_half-disp:x+kernel_half+1-disp,:].reshape(-1) > l_right_center.reshape(3,1)
                    l_combi_cencus = np.logical_xor(l_tmp_left, l_tmp_right)

                if x+kernel_half+1+disp < w:
                    if x+kernel_half+disp < w:
                        r_tmp_left = Il[y-kernel_half:y+kernel_half+1, x-kernel_half+disp:x+kernel_half+1+disp,:].reshape(-1) > r_left_center.reshape(3,1)
                    else:
                        r_tmp_left = Il[y-kernel_half:y+kernel_half+1, x-kernel_half+disp-w:x+kernel_half+1+disp-w,:].reshape(-1) > r_left_center.reshape(3,1)
                    r_tmp_right = Ir[y-kernel_half:y+kernel_half+1, x-kernel_half:x+kernel_half+1,:].reshape(-1) > r_right_center.reshape(3,1)
                    r_combi_cencus = np.logical_xor(r_tmp_right, r_tmp_left)


                # count = 0
                # for v in range(-kernel_half, kernel_half):
                #     for u in range(-kernel_half, kernel_half):
                #         # Census Transform
                #         l_tmp_left = l_left_center < Il[y+v, x+u]
                #         l_tmp_right = l_right_center < Ir[y+v, (x+u)-disp]
                #         l_combi_cencus[count] = np.logical_xor(l_tmp_left, l_tmp_right)
                #
                #         # right
                #         if x+u+disp < w:
                #             r_tmp_left = r_left_center < Il[y+v, x+u+disp]
                #         else:
                #             r_tmp_left = r_left_center < Il[y+v, x+u+disp-w]
                #         r_tmp_right = r_right_center < Ir[y+v, (x+u)]
                #         r_combi_cencus[count] = np.logical_xor(r_tmp_right, r_tmp_left)
                #         count += 1

                left_matching_cost[y-kernel_half, x-kernel_half, i_disp] = np.sum(l_combi_cencus)
                right_matching_cost[y-kernel_half, x-kernel_half, i_disp] = np.sum(r_combi_cencus)

    # >>> Cost aggregation
    # TODO: Refine cost by aggregate nearby costs
    # Guided image filter
    for i_disp, disp in enumerate(range(max_disp)):
        left_matching_cost[:,:,i_disp] = cv2.blur(left_matching_cost[:,:,i_disp],(5,5))
        right_matching_cost[:,:,i_disp] = cv2.blur(right_matching_cost[:,:,i_disp],(5,5))


    # >>> Disparity optimization
    # TODO: Find optimal disparity based on estimated cost. Usually winner-take-all.
    left_labels = left_matching_cost.argmin(2) #* offset_adjust
    right_labels = right_matching_cost.argmin(2) #* offset_adjust

    # >>> Disparity refinement
    # TODO: Do whatever to enhance the disparity map
    # ex: Left-right consistency check + hole filling + weighted median filtering
    left_labels = left_labels.astype(np.uint8)
    right_labels = right_labels.astype(np.uint8)
    # left_labels = cv2.medianBlur(left_labels, 5)
    # right_labels = cv2.medianBlur(right_labels, 5)

    # cv2.imwrite('L.png', np.uint8(left_labels*16))
    # cv2.imwrite('R.png', np.uint8(left_labels*16))
    # cv2.imwrite('tsukuba_r.png', np.uint8(right_labels * 16))
    # Left-right consistency check
    left_labels = LR_consistency_check(left_labels, right_labels, max_disp)

    # median filter (Ref: https://blog.csdn.net/streamchuanxi/article/details/79573302)
    left_labels = cv2.medianBlur(left_labels, 23)

    #cv.ximgproc.weightedMedianFilter()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(12,12))
    left_labels = cv2.morphologyEx(left_labels,cv2.MORPH_OPEN,kernel)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    # left_labels = cv2.morphologyEx(left_labels,cv2.MORPH_OPEN,kernel)
    #
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    # left_labels = cv2.morphologyEx(left_labels,cv2.MORPH_OPEN,kernel)

    left_labels = cv2.bilateralFilter(left_labels, 10, 9, 16)

    return left_labels.astype(np.uint8)

def LR_consistency_check(left_labels, right_labels, max_disp):

    assert left_labels.shape == right_labels.shape
    h, w = left_labels.shape
    valid_map = np.zeros((h, w)).astype(np.int)

    new_labels = left_labels.copy()
    # find invalid part by Left-right consistency check
    for j in range(h):
        for i in range(w):
            left_disparity = left_labels[j, i]
            right_disparity = right_labels[j, i-left_disparity]
            if i-left_disparity < 0: # out off boundary
                valid_map[j, i] = 0
            else:
                if left_disparity == right_disparity:
                    valid_map[j, i] = 1
                else:
                    valid_map[j, i] = 0

    # now start to match the invalid pattern
    refine_threshold = max_disp // 6
    for j in range(h):
        for i in range(w):
            if valid_map[j, i]:
                pass
            else:
                disparity = 1e6
                left_valid_dist = 0
                right_valid_dist = 0
                for left_idx in [idx for idx in range(i)] + [i]:
                    if valid_map[j, i-left_idx] and left_labels[j, i-left_idx] > refine_threshold:
                        disparity = min(disparity, left_labels[j, i-left_idx])
                        break
                for right_idx in [idx for idx in range(w-i-1)] + [w-i-1]:
                    if valid_map[j, i+right_idx] and left_labels[j, i+right_idx] > refine_threshold:
                        # right_valid_dist = right_idx
                        disparity = min(disparity, left_labels[j, i+right_idx])
                        break
                new_labels[j, i] = disparity
                # modify left disparity
                # if left_valid_dist == 0:
                #     new_labels[j, i] = left_labels[j, i+right_valid_dist]
                # elif right_valid_dist == 0:
                #     new_labels[j, i] = left_labels[j, i-left_valid_dist]
                # else:
                #     if left_valid_dist < right_valid_dist:
                #         new_labels[j, i] = left_labels[j, i-left_valid_dist]
                #     else:
                #         new_labels[j, i] = left_labels[j, i+right_valid_dist]

    return new_labels.astype(np.uint8)

def box_filter(input_data, box):
    h, w, ch = input_data.shape
    b_h, b_w = box.shape
    kernel_half = int(b_h / 2)
    for y in tqdm(range(kernel_half, h + kernel_half)):
        for x in range(kernel_half, w + kernel_half):
            pass
    return input_data
