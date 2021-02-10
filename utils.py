import torch
from torch import nn
import numpy as np
import cv2


def compute_probability_of_activation(result, roi, threshold):
    '''
    results: numpy array of shape (h, w)
    roi: numpy array of shape (4) (xyxy)
    '''
    # roi = [b.item() for b in roi]
    result = (result > threshold) * result
    activation_ratio = result[:, roi[1]:roi[3], roi[0]:roi[2]].sum() / ((roi[3] - roi[1]) * (roi[2] - roi[0]))
    return activation_ratio > threshold


def compute_probability_of_activations(results, rois, threshold):
    total_length = len(results)
    bool_results = 0
    for result, roi in zip(results, rois):        
        bool_results += compute_probability_of_activation(result, roi, threshold)
    return bool_results / total_length


def save_img(activation, image, box, path):
    '''
    activation: numpy array 0~255
    '''
    heatmap = cv2.applyColorMap(np.uint8(activation), cv2.COLORMAP_JET)
    # draw box to heatmap
    heatmap = np.float32(heatmap) / 255
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    # img = cv2.resize(img, (384, 384))#(heatmap.shape[1], heatmap.shape[0]))

    cam = heatmap + np.float32(image) / 255
    cam = cam / np.max(cam)

    # h_ratio = img.shape[0] / o.shape[0]
    # w_ratio = img.shape[1] / o.shape[1]
    # xyxy
    # box = [b.item() for b in boxes[i]]
    box = tuple([int(b) for b in box])
    cv2.rectangle(cam, box[:2], box[2:], (255,255,0), 2)
    cam = cam * 255
    cam = np.concatenate([cam, image], axis=1)
    cv2.imwrite(path, cam.astype(np.uint8))
    print('saved: ', path)