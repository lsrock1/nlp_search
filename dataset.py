#!/usr/bin/env python
# COPYRIGHT 2020. Fred Fung. Boston University.
"""
PyTorch dataset for CityFlow-NL.
"""
import json
import os
import random
import numpy as np
from collections import defaultdict
import pickle

import cv2
import torch
from torch.utils.data import Dataset
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords


class NL:
    def __init__(self, cfg, tracks):
        self.cfg = cfg
        self.tracks = tracks
        self.s = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

        if os.path.exists(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl')):
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_count.pkl'), 'rb') as handle:
                self.words_count = pickle.load(handle)
            with open(os.path.join(self.cfg.DATA.DICT_PATH, 'word_to_idx.pkl'), 'rb') as handle:
                self.word_to_idx = pickle.load(handle)
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

        for t in tracks:
            for n in t['nl']:
                for w in self.do_clean(n):
                # for l in n.replace('.', '').split():
                    word_count[w] += 1
        
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

        return word_count, word_to_idx

    def do_clean(self, nl):
        nl = nl[:-1].lower().split()
        nl = [self.s.stem(w) for w in nl]
        nl = [w for w in nl if w not in self.stop_words]
        return nl

    def sentence_to_index(self, nl):
        nl = self.do_clean(nl)
        idxs = [self.word_to_idx[n] if n in self.word_to_idx else self.word_to_idx['<UNK>'] for n in nl]
        
        if len(idxs) > self.cfg.DATA.MAX_SENTENCE:
            idxs = idxs[:self.cfg.DATA.MAX_SENTENCE]
            idxs = [self.word_to_idx['<SOS>']] + idxs + [self.word_to_idx['<EOS>']]
        else:
            idxs = [self.word_to_idx['<SOS>']] + idxs + [self.word_to_idx['<EOS>']] + [self.word_to_idx['<PAD>'] for _ in range(self.cfg.DATA.MAX_SENTENCE - len(idxs))]
        return idxs


class CityFlowNLDataset(Dataset):
    def __init__(self, data_cfg, transforms):
        """
        Dataset for training.
        :param data_cfg: CfgNode for CityFlow NL.
        """
        self.data_cfg = data_cfg.clone()
        with open(self.data_cfg.DATA.JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.list_of_crops = list()
        for track in self.list_of_tracks:
            for frame_idx, frame in enumerate(track["frames"]):
                if not os.path.exists(os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame)):
                    # print('not exists', os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame))
                    continue
                frame_path = os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame)
                nl_idx = int(random.uniform(0, 3))
                nl = track["nl"][nl_idx]
                box = track["boxes"][frame_idx]
                crop = {"frame": frame_path, "nl": nl, "box": box}
                self.list_of_crops.append(crop)
        # new_list_of_tracks = []
        # for track in self.list_of_tracks:
        #     new_frames = []
        #     new_boxes = []
        #     for f, b in zip(track['frames'], track['boxes']):
        #         if os.path.exists(os.path.join(data_cfg.DATA.CITYFLOW_PATH, f)):
        #             new_frames.append(f)
        #             new_boxes.append(b)
        #     if len(new_frames) > 0:
        #         # print(len(new_frames))
        #         new_list_of_tracks.append({
        #             'frames': new_frames,
        #             'boxes': new_boxes,
        #             'nl': track['nl']
        #         })
        # self.list_of_tracks = new_list_of_tracks
        self.nl = NL(data_cfg, self.list_of_tracks)
        self.transforms = transforms

    def __len__(self):
        return len(self.list_of_crops)

    def __getitem__(self, index):
        """
        Get pairs of NL and cropped frame.
        """
        dp = self.list_of_crops[index]
        frame = cv2.imread(dp["frame"])
        h, w, _ = frame.shape
        frame = torch.from_numpy(frame).permute([2, 0, 1])
        box = dp["box"]
        nl = self.nl.sentence_to_index(dp["nl"])
        ymin, ymax = box[1], box[1] + box[3]
        xmin, xmax = box[0], box[0] + box[2]
        frame = self.transforms[0](frame)
        
        h_ratio = self.data_cfg.DATA.GLOBAL_SIZE[0] / h
        w_ratio = self.data_cfg.DATA.GLOBAL_SIZE[1] / w
        ymin, ymax = int(ymin * h_ratio / 16), int(ymax * h_ratio / 16)
        xmin, xmax = int(xmin * w_ratio / 16), int(xmax * w_ratio / 16)
        # print(xmin, ' ', xmax, ' ', ymin, ' ', ymax)
        # crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
        # crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
        # frame = frame.permute([2, 0, 1])
        label = torch.zeros([1, self.data_cfg.DATA.GLOBAL_SIZE[0]//16, self.data_cfg.DATA.GLOBAL_SIZE[1]//16])
        label[:, ymin:ymax, xmin:xmax] = 1
        # print(label.shape)
        # print(frame.shape)
        # print(len(nl))
        # track = self.list_of_tracks[index]
        # frames = track["frames"]
        # frames_idx = np.random.choice(len(frames), self.data_cfg.DATA.NUM_IMG)
        # # print(frames_idx)
        # selected_frames = [frames[i] for i in frames_idx]
        # # print(selected_frames)
        # selected_boxes = np.array(track["boxes"])[frames_idx]
        # nl = track["nl"][random.randrange(3)]
        # nl = self.nl.sentence_to_index(nl)
        # # print([os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, s) for s in selected_frames])
        # selected_frames = [cv2.imread(os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, s)) for s in selected_frames]
        # # print(selected_frames)
        # selected_boxes = [s[b[1]:b[1]+b[3], b[0]:b[0]+b[2], :] for b, s in zip(selected_boxes, selected_frames) ]
        
        # selected_frames = [torch.from_numpy(s).permute([2, 0, 1]) for s in selected_frames]
        # selected_boxes = [torch.from_numpy(s).permute([2, 0, 1]) for s in selected_boxes]
        
        # selected_frames = torch.stack([self.transforms[0](s) for s in selected_frames], dim=0)
        # selected_boxes = torch.stack([self.transforms[1](s) for s in selected_boxes], dim=0)

        # frame_path = os.path.join(self.data_cfg.CITYFLOW_PATH, frame)
        # frame = cv2.imread(dp["frame"])
        # box = dp["box"]
        # crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
        # crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
        # crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
        #     dtype=torch.float32)

        return torch.tensor(nl), frame, label


class CityFlowNLInferenceDataset(Dataset):
    def __init__(self, data_cfg, transforms):
        """Dataset for evaluation. Loading tracks instead of frames."""
        self.data_cfg = data_cfg
        with open(self.data_cfg.DATA.EVAL_TRACKS_JSON_PATH) as f:
            tracks = json.load(f)
        self.list_of_uuids = list(tracks.keys())
        self.list_of_tracks = list(tracks.values())
        self.transforms = transforms
        self.nl = NL(data_cfg, self.list_of_tracks)

    def __len__(self):
        return len(self.list_of_uuids)

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
        for frame_path, box in zip(dp["frames"], dp["boxes"]):
            frame_path = os.path.join(self.data_cfg.DATA.CITYFLOW_PATH, frame_path)
            if not os.path.isfile(frame_path):
                continue
            paths.append(frame_path)
            frame = cv2.imread(frame_path)
            h, w, _ = frame.shape
            frame = self.transforms[0](frame)
            frames.append(frame)
            # boxes.append(box)

            ymin, ymax = box[1], box[1] + box[3]
            xmin, xmax = box[0], box[0] + box[2]
            # frame = self.transforms[0](frame)
            boxes.append([xmin, ymin, xmax, ymax])
            h_ratio = self.data_cfg.DATA.GLOBAL_SIZE[0] / h
            w_ratio = self.data_cfg.DATA.GLOBAL_SIZE[1] / w
            ymin, ymax = int(ymin * h_ratio / 16), int(ymax * h_ratio / 16)
            xmin, xmax = int(xmin * w_ratio / 16), int(xmax * w_ratio / 16)
            rois.append([xmin, ymin, xmax, ymax])
            # box = dp["boxes"][frame_idx]
            # crop = frame[box[1]:box[1] + box[3], box[0]: box[0] + box[2], :]
            # crop = cv2.resize(crop, dsize=self.data_cfg.CROP_SIZE)
            # crop = torch.from_numpy(crop).permute([2, 0, 1]).to(
                # dtype=torch.float32)
            # cropped_frames.append(crop)
        # dp["crops"] = torch.stack(cropped_frames, dim=0)
        frames = torch.stack(frames, dim=0)
        return id, frames, boxes, paths, rois


def query(data_cfg):
    with open(data_cfg.DATA.EVAL_QUERIES_JSON_PATH) as f:
        tracks = json.load(f)
    uuids = list(tracks.keys())
    nls = list(tracks.values())
    return uuids, nls