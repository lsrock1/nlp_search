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
    return activation_ratio


def compute_probability_of_point(result, roi, threshold):
    '''
    results: numpy array of shape (h, w)
    roi: numpy array of shape (4) (xyxy)
    '''
    # print(result.shape)
    # print(result[:, (roi[1] + roi[3])//2, (roi[0] + roi[2])//2].shape)
    return result[:, (roi[1] + roi[3])//2, (roi[0] + roi[2])//2].item()# > threshold


def compute_probability_of_activations(results, rois, threshold):
    total_length = len(results)
    bool_results = 0
    for result, roi in zip(results, rois):        
        activation_ratio = compute_probability_of_activation(result, roi, threshold)
        bool_results += activation_ratio# > threshold
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


def stableMatching(matrix):
    # Initially, all n men are unmarried
    num_nl, num_img = matrix.shape
    menPreferences = (-matrix).argsort(axis=1).tolist()
    womenPreferences = np.transpose((-matrix).argsort(axis=0)).tolist()
    unmarriedMen = list(range(num_nl))
    # None of the men has a spouse yet, we denote this by the value None
    manSpouse = [None] * num_nl                      
    # None of the women has a spouse yet, we denote this by the value None
    womanSpouse = [None] * num_img                      
    # Each man made 0 proposals, which means that 
    # his next proposal will be to the woman number 0 in his list
    nextManChoice = [0] * num_nl                       
    
    # While there exists at least one unmarried man:
    while unmarriedMen:
        # Pick an arbitrary unmarried man
        he = unmarriedMen[0]                      
        # Store his ranking in this variable for convenience
        hisPreferences = menPreferences[he]       
        # Find a woman to propose to
        she = hisPreferences[nextManChoice[he]] 
        # Store her ranking in this variable for convenience
        herPreferences = womenPreferences[she]
        # Find the present husband of the selected woman (it might be None)
        currentHusband = womanSpouse[she]
       
        
        # Now "he" proposes to "she". 
        # Decide whether "she" accepts, and update the following fields
        # 1. manSpouse
        # 2. womanSpouse
        # 3. unmarriedMen
        # 4. nextManChoice
        if currentHusband == None:
            #No Husband case
            #"She" accepts any proposal
            womanSpouse[she] = he
            manSpouse[he] = she
            #"His" nextchoice is the next woman
            #in the hisPreferences list
            nextManChoice[he] = nextManChoice[he] + 1
            #Delete "him" from the 
            #Unmarried list
            unmarriedMen.pop(0)
        else:
            #Husband exists
            #Check the preferences of the 
            #current husband and that of the proposed man's
            currentIndex = herPreferences.index(currentHusband)
            hisIndex = herPreferences.index(he)
            #Accept the proposal if 
            #"he" has higher preference in the herPreference list
            if currentIndex > hisIndex:
                #New stable match is found for "her"
                womanSpouse[she] = he
                manSpouse[he] = she
                nextManChoice[he] = nextManChoice[he] + 1
                #Pop the newly wed husband
                unmarriedMen.pop(0)
                #Now the previous husband is unmarried add
                #him to the unmarried list
                unmarriedMen.insert(0,currentHusband)
            else:
                nextManChoice[he] = nextManChoice[he] + 1
    return manSpouse