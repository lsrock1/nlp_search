#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
PyTorch dataset for CityFlow-NL.
"""
import json
import os
import random
import numpy as np
import pickle
from collections import Counter, defaultdict
import itertools

import cv2
import torch
from torch.utils.data import Dataset
from nltk.stem import PorterStemmer
import albumentations as A
import albumentations.pytorch as AP
from nltk.corpus import stopwords
from transformers import ElectraTokenizerFast
# import nltk
# nltk.download('stopwords')

class NL:
    def __init__(self, cfg, tracks):
        self.cfg = cfg
        self.tracks = tracks
        self.s = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tokenizer = ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')

        if os.path.exists(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl')):
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl'), 'rb') as handle:
                self.words_count = pickle.load(handle)
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_to_idx.pkl'), 'rb') as handle:
                self.word_to_idx = pickle.load(handle)
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'special_case.pkl'), 'rb') as handle:
                self.special_case = pickle.load(handle)
        else:
            self.words_count, self.word_to_idx = self.__build_dict(self.tracks)

    def __len__(self):
        return len(self.word_to_idx)

    def __build_dict(self, tracks):
        word_count = defaultdict(int)

        word_count['<SOS>'] += 1
        word_count['<EOS>'] += 1
        word_count['<PAD>'] += 1
        word_count['<UNK>'] += 1
        max_length = 0

        # special case handling
        except_case = ['dark-red', 'dark-blue', 'dark-colored']
        hand_case = {
            'hatckback': 'hatchback'
        }
        self.special_case = {}
        for t in tracks:
            for n in t['nl']:
                for word in n.lower()[:-1].split():
                    if '-' in word and word not in except_case and word not in self.special_case:
                        self.special_case[word.replace('-', ' ')] = word.replace('-', '')
        
        for ec in except_case:
            self.special_case[ec] = ec.replace('-', ' ')
        self.special_case.update(hand_case)
        
        # self.special_case = special_case
                        
        for t in tracks:
            for n in t['nl']:
                cleaned_sentence = self.do_clean(n)
                if len(cleaned_sentence) > max_length:
                    max_length = len(cleaned_sentence)
                for w in cleaned_sentence:
                # for l in n.replace('.', '').split():
                    word_count[w] += 1
        print('max: ', max_length)
        new_dict = dict()
        for k, v in word_count.items():
            if v >= self.cfg.DATA.MIN_COUNT or k in ['<SOS>', '<EOS>', '<PAD>', '<UNK>']:
                new_dict[k] = v
        
        word_count = new_dict

        word_to_idx = dict(zip(word_count.keys(), range(len(word_count))))

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl'), 'wb') as handle:
            pickle.dump(word_count, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_to_idx.pkl'), 'wb') as handle:
            pickle.dump(word_to_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(self.cfg.DATA.DICT_PATH, 'special_case.pkl'), 'wb') as handle:
            pickle.dump(self.special_case, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        return word_count, word_to_idx

    def do_clean(self, nl):
        nl = nl.lower()
        for sc, replaced in self.special_case.items():
            if sc in nl:
                nl = nl.replace(sc, replaced)

        nl = nl[:-1].replace('-', '').split()
        # nl = [self.s.stem(w) for w in nl]
        # nl = [w for w in nl if w not in self.stop_words]
        return nl

    def sentence_to_index(self, nl, is_train=True):
        nl = self.do_clean(nl)
        str_input = ' '.join(nl)
        if is_train:
            idxs = self.tokenizer.encode(str_input, padding='max_length', truncation=True, max_length=30)
        else:
            idxs = self.tokenizer.encode(str_input)
        return idxs


class CityFlowNLDataset(Dataset):
    colors = ['silver', 'red', 'white', 'brown', 'gold','black', 'gray', 'blue', 'purple', 'yellow', 'orange', 'green']
    vehicle_type = ['bus', 'pickup','sedan', 'suv', 'van', 'wagon', 'cargo', 'mpv', 'hatchback', 'coup','truck', 'minivan']

    def __init__(self, data_cfg, transforms):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        # self.vehicle_type = ['bus', 'pickup','sedan', 'suv', 'van', 'wagon', 'cargo', 'mpv', 'hatchback', 'coup','truck', 'minivan']
        # self.colors = ['silver', 'red', 'white', 'brown', 'gold','black', 'gray', 'blue', 'purple', 'yellow', 'orange', 'green']
        with open(self.data_cfg.DATA.JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        self.nl = NL(data_cfg, self.list_of_tracks)
        self.color_type_item = dict([[(c, t), []] for c in range(len(CityFlowNLDataset.colors)) for t in range(len(CityFlowNLDataset.vehicle_type)-2)])
        # self.color_type_list = [[[] for _ in range(len(CityFlowNLDataset.vehicle_type)-2)] for _ in range(len(CityFlowNLDataset.colors))]
        self.color_type_per_frame = defaultdict(set)
        # self.type_per_frame = defaultdict(set)
        # if os.path.exists(os.path.join(self.data_cfg.DATA.DICT_PATH, 'list_of_crops.pkl')):
        #     with open(os.path.join(self.data_cfg.DATA.DICT_PATH, 'list_of_crops.pkl'), 'rb') as handle:
        #         self.list_of_crops = pickle.load(handle)
        #     # self.list_of_crops = 
        # else:
        for track_idx, track in enumerate(self.list_of_tracks):
            # print(track_idx, '/', len(self.list_of_tracks))
            for frame_idx, frame in enumerate(track["frames"]):
                if not os.path.exists(os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame)):
                    # print(os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame))
                    # print('not exists', os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame))
                    continue
                frame_path = os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame)
                #nl_idx = int(random.uniform(0, 3))
                nl = track["nl"]#[nl_idx]
                box = track["boxes"][frame_idx]
                # crop = {"frame": frame_path, "nl": nl, "box": box}
                # self.list_of_crops.append(crop)
                # expand nls
                nl, vehicle_type = CityFlowNLDataset.type_replacer(nl)
                if vehicle_type == -1:
                    continue
                nl, vehicle_color = CityFlowNLDataset.color_replacer(nl)
                if vehicle_color == -1:
                    continue
                # print(len(nl))
                # print(nl)
                self.color_type_per_frame[frame_path].add((vehicle_color, vehicle_type))
                # self.type_per_frame[frame_path].add(vehicle_type)
                for n in nl:
                    crop = {"frame": frame_path, "nl": n, "box": box, "color": vehicle_color, "type": vehicle_type}
                    self.list_of_crops.append(crop)
                    self.color_type_item[(vehicle_color, vehicle_type)].append(
                        len(self.list_of_crops) - 1
                    )
            # with open(os.path.join(self.data_cfg.DATA.DICT_PATH, 'list_of_crops.pkl'), 'wb') as handle:
            #     pickle.dump(self.list_of_crops, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.transforms = transforms
        print('data loading end')

    def __len__(self):
        return len(self.list_of_crops)

    @classmethod
    def color_replacer(cls, nls):
        # cleaning noise in nl
        color_replace = {
            'reddish': 'red',
            'maroon': 'red', 'whit ': 'white ', 'golden': 'gold', 'grey' :'gray', 'lightgray': 'gray'
        }
        
        colors_in_nls = []
        replace_list = []
        nls = [n.lower() for n in nls]
        new_nls = []
        for nl in nls:
            for k, v in color_replace.items():
                if k in nl:
                    nl = nl.replace(k, v)
            new_nls.append(nl)
        nls = new_nls
        for nl in nls:
            # n = n.lower()
            colors_in_nl = []
            
            for c in cls.colors:
                index = nl.find(c)
                if index != -1:
                    # colors_in_nls.add(c)
                    colors_in_nl.append((c, index))
            colors_in_nl = sorted(colors_in_nl, key=lambda x: x[1])
            if len(colors_in_nl) > 0:
                colors_in_nls.append(colors_in_nl[0][0])
                replace_list.append(colors_in_nl[0][0])
            else:
                replace_list.append(None)
        
        if len(set(colors_in_nls)) > 1:
            counters = Counter(colors_in_nls)
            # nls = [t.lower() for t in track['nl']]
            k = counters.most_common(1)
            representer_color = k[0][0]
            new_nls = []
            for c, n in zip(replace_list, nls):
                if c != None:
                    new_nls.append(n.replace(c, representer_color))
                else:
                    new_nls.append(n)
            return new_nls, cls.colors.index(representer_color)
        elif len(colors_in_nls) == 0:
            return nls, -1
        return nls, cls.colors.index(colors_in_nls[0])

    @classmethod
    def type_replacer(cls, nls):
        # cleaning noise in nl
        type_replace = {'hatckback': 'hatchback'}
        types_in_nls = []
        replace_list = []
        nls = [n.lower() for n in nls]
        new_nls = []
        for nl in nls:
            for k, v in type_replace.items():
                if k in nl:
                    nl = nl.replace(k, v)
                # else:
                #     new_nls.append(nl)
            new_nls.append(nl)
        nls = new_nls
        for nl in nls:
            types_in_nl = []
            
            for c in cls.vehicle_type:
                index = nl.find(c)
                if index != -1:
                    # colors_in_nls.add(c)
                    types_in_nl.append((c, index))
            types_in_nl = sorted(types_in_nl, key=lambda x: x[1])

            if len(types_in_nl) > 0:
                first_type = types_in_nl[0][0]
                if first_type == 'truck':
                    first_type = 'pickup'
                elif first_type == 'minivan':
                    first_type = 'van'
                types_in_nls.append(first_type)
                replace_list.append(first_type)
            else:
                replace_list.append(None)
        
        if len(set(types_in_nls)) > 1:
            counters = Counter(types_in_nls)
            # nls = [t.lower() for t in track['nl']]
            k = counters.most_common(1)
            representer_color = k[0][0]
            new_nls = []
            for c, n in zip(replace_list, nls):
                if c != None:
                    new_nls.append(n.replace(c, representer_color))
                else:
                    new_nls.append(n)
            return new_nls, cls.vehicle_type.index(representer_color)
        elif len(types_in_nls) == 0:
            return nls, -1
        return nls, cls.vehicle_type.index(types_in_nls[0])

    def bbox_aug(self, img, bbox, h, w):
        resized_h = int(h * 0.8)
        resized_w = int(w * 0.8)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
        first = [max(xmax - resized_w, 0), max(ymax - resized_h, 0)]
        second = [min(xmin + resized_w, w) - resized_w, min(ymin + resized_h, h) - resized_h]
        if first[0] > second[0] or first[1] > second[1]:
            tf = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Resize(self.data_cfg.DATA.GLOBAL_SIZE[0], self.data_cfg.DATA.GLOBAL_SIZE[1]),
                    AP.transforms.ToTensor(normalize={
                        'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]
                    })
                ],
                bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
            )(image=img, bboxes=[bbox], class_labels=[0])
            # print(tf['bboxes'])
            return tf['image'], tf['bboxes'][0]
            # return img, bbox
        x = random.randint(first[0], second[0])
        y = random.randint(first[1], second[1])

        # print(bbox)
        tf = A.Compose(
            [
                A.Crop(x_min=x, y_min=y, x_max=x+resized_w, y_max=y+resized_h, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Resize(self.data_cfg.DATA.GLOBAL_SIZE[0], self.data_cfg.DATA.GLOBAL_SIZE[1]),
                AP.transforms.ToTensor(normalize={
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                })
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
        )(image=img, bboxes=[bbox], class_labels=[0])
        # print(tf['bboxes'])
        return tf['image'], tf['bboxes'][0]

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        dp = self.list_of_crops[index]
        frame = cv2.imread(dp["frame"])
        h, w, _ = frame.shape
        box = dp["box"]
        frame, box = self.bbox_aug(frame, box, h, w)
        color, typ = dp['color'], dp['type']

        ymin, ymax = box[1], box[1] + box[3]
        xmin, xmax = box[0], box[0] + box[2]
        xmin, xmax, ymin, ymax = int(xmin//16), int(xmax//16), int(ymin//16), int(ymax//16)

        label = torch.zeros([1, self.data_cfg.DATA.GLOBAL_SIZE[0]//16, self.data_cfg.DATA.GLOBAL_SIZE[1]//16])
        # rectangle version
        label[:, ymin:ymax, xmin:xmax] = 1

        if random.random() >= 0.8:# and len(self.color_type_list[color][typ]) > 1:
            # different type same color
            selected_items = []
            for combination, l in self.color_type_item.items():
                if combination not in self.color_type_per_frame[dp["frame"]] and (combination[0] == color or combination[1] == typ):
                    # print(f"{color}, {typ}", ' ', combination)
                    selected_items.append(l)
            # dtsc = [self.color_type_list[color][t] for t in range(len(self.color_type_list[color])) if t not in self.color_type_per_frame[dp['frame']][color] and len(self.color_type_list[color][t]) > 1]
            # same type different color
            # stdc = [self.color_type_list[c][typ] for c in range(len(self.color_type_list)) if typ not in self.color_type_per_frame[dp['frame']][c] and len(self.color_type_list[c][typ]) > 1]
            new_dp_idx = random.choice(list(itertools.chain(*selected_items)))
            
            dp = self.list_of_crops[new_dp_idx]
            nl = dp["nl"]#[int(random.uniform(0, 3))]
            nl = self.nl.sentence_to_index(nl)
            label_ = torch.zeros([1, self.data_cfg.DATA.GLOBAL_SIZE[0]//16, self.data_cfg.DATA.GLOBAL_SIZE[1]//16])

            return torch.tensor(nl), frame, label_, label, color, typ, dp['color'], dp['type']
        
        nl = dp["nl"]#[0][int(random.uniform(0, 3))]
        # print(nl)
        nl = self.nl.sentence_to_index(nl)
        
        return torch.tensor(nl), frame, label, label, color, typ, color, typ


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg, transforms, num_frames=None):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        with open(self.data_cfg.DATA.EVAL_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transforms = transforms
        self.nl = NL(data_cfg, self.list_of_tracks)
        self.load_frame = True
        self.num_frames = num_frames

    def __len__(self):
        return len(self.list_of_uuids)

    def bbox_aug(self, img, bbox):
        tf = A.Compose(
            [
                A.Resize(self.data_cfg.DATA.GLOBAL_SIZE[0], self.data_cfg.DATA.GLOBAL_SIZE[1]),
                AP.transforms.ToTensor(normalize={
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                })
            ],
            bbox_params=A.BboxParams(format='coco', label_fields=['class_labels']),
        )(image=img, bboxes=[bbox], class_labels=[0])
        # print(tf['bboxes'])
        return tf['image'], tf['bboxes'][0]

    def __getitem__(self, index):
        """
        :return: a dictionary for each track:
        id: uuid for the track
        frames, boxes, nl are untouched from the input json file.
        crops: A Tensor of cropped images from the track of shape
            [length, 3, crop_w, crop_h].
        """
        id = self.list_of_uuids[index]
        dp = self.list_of_tracks[index]
        # nl = dp['nl']
        # nl = self.nl.sentence_to_index(nl)
        # dp = {"id": self.list_of_uuids[index]}
        # dp.update(self.list_of_tracks[index])
        frames = []
        boxes = []
        paths = []
        rois = []
        labels = []
        for idx, (frame_path, box) in enumerate(zip(dp["frames"], dp["boxes"])):
            if self.num_frames != None and len(frames) == self.num_frames:
                break
            frame_path = os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame_path)
            if not os.path.isfile(frame_path):
                continue
            paths.append(frame_path)

            if self.load_frame:
                frame = cv2.imread(frame_path)
                h, w, _ = frame.shape
                frame, box_resized = self.bbox_aug(frame, box)
                frames.append(frame)
            else:
                if idx == 0:
                    frame = cv2.imread(frame_path)
                    h, w, _ = frame.shape
                frames.append(torch.zeros(1))
            # boxes.append(box)

            ymin, ymax = box[1], box[1] + box[3]
            xmin, xmax = box[0], box[0] + box[2]
            # frame = self.transforms[0](frame)
            boxes.append([xmin, ymin, xmax, ymax])
            h_ratio = self.data_cfg.DATA.GLOBAL_SIZE[0] / h
            w_ratio = self.data_cfg.DATA.GLOBAL_SIZE[1] / w
            ymin, ymax = int(ymin * h_ratio // 16), int(ymax * h_ratio // 16)
            xmin, xmax = int(xmin * w_ratio // 16), int(xmax * w_ratio // 16)
            rois.append([xmin, ymin, xmax, ymax])
            label = torch.zeros([1, self.data_cfg.DATA.GLOBAL_SIZE[0]//16, self.data_cfg.DATA.GLOBAL_SIZE[1]//16])
            # rectangle version
            label[:, ymin:ymax, xmin:xmax] = 1
            labels.append(label)
            # box = dp["boxes"][frame_idx]
            # crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
            # crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
            # crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
                # dtype=torch.float32)
            # cropped_frames.append(crop)
        # dp["crops"] = torch.stack(cropped_frames, dim=0)
        frames = torch.stack(frames, dim=0)
        return id, frames, np.array(boxes), paths, np.array(rois), torch.stack(labels, dim=0)


def query(data_cfg):
    with open(data_cfg.DATA.EVAL_QUERIES_JSON_PATH) as f:
        tracks = json.load(f)
    uuids = list(tracks.keys())
    nls = list(tracks.values())
    return uuids, nls