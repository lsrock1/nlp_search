from dataset import CityFlowNLDataset, CityFlowNLInferenceDataset, query
from configs import get_default_config
from model import MyModel
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss
from utils import compute_probability_of_activations, save_img

from torch.nn.parallel import DistributedDataParallel
from torch import nn
from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import MultiStepLR
import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm
from glob import glob
import json

epoch = 19
test_batch_size = 64
scene_threshold = 0.8
total_threshold = 0.5
num_of_vehicles = 64

cfg = get_default_config()
dataset = CityFlowNLInferenceDataset(cfg, build_transforms(cfg), num_of_vehicles)
model = MyModel(cfg, len(dataset.nl), dataset.nl.word_to_idx['<PAD>'], nn.BatchNorm2d, num_colors=len(CityFlowNLDataset.colors), num_types=len(CityFlowNLDataset.vehicle_type) - 2).cuda()

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
uuids, nls = query(cfg)

saved_dict = torch.load(f'save/{epoch}.pth')

n = {}
for k, v in saved_dict.items():
    n[k.replace('module.', '')] = v

model.load_state_dict(n, False)
model.eval()

if os.path.exists('results'):
    shutil.rmtree('results')
os.mkdir('results')

# extract img fts first to save time
if not os.path.exists('cache'):
    # shutil.rmtree('cache')
    os.mkdir('cache')

if not os.path.exists(f'cache/{epoch}'):
    os.mkdir(f'cache/{epoch}')

    with torch.no_grad():
        for idx, (id, frames, boxes, paths, rois, _) in enumerate(tqdm(loader)):
            frames = frames.squeeze(0).cuda()
            b = frames.shape[0]
            cache = []

            # version 3
            # if b <= test_batch_size:
            #     cache = model.cnn(frames)
            #     torch.save(cache, f'cache/{epoch}/{idx}_0.pth')
            # else:
            #     cache = []
            for i, f in enumerate(frames.split(test_batch_size)):
                cache = model.cnn(f)
                torch.save(cache, f'cache/{epoch}/{idx}_{i}.pth')

        print('saving language features..')
        for uuid, query_nl in zip(uuids, nls):
            nls_list = []
            query_nl, vehicle_type = CityFlowNLDataset.type_replacer(query_nl)
            query_nl, vehicle_color = CityFlowNLDataset.color_replacer(query_nl)
            # max_len = max([len(dataset.nl.do_clean(nl)) for nl in query_nl])
            for nl in query_nl:
                nl = torch.tensor(dataset.nl.sentence_to_index(nl, is_train=False)).cuda()
                # nls.append(nl.unsqueeze(0).transpose(1, 0))
                nl = nl.unsqueeze(0).transpose(1, 0)
                # bs, len, dim
                nl = model.rnn(nl)
                nls_list.append(nl)
            saved_nls = {
                'nls': nls_list,
                'type': vehicle_type, 'color': vehicle_color
            }
            torch.save(saved_nls, f'cache/{epoch}/{uuid}.pth')

dataset.load_frame = False

# model = nn.DataParallel(model)
final_results = {}
for nlidx, (uuid, query_nl) in enumerate(zip(uuids, nls)):
    print(f'{nlidx} / {len(nls)}')
    cache_nl = torch.load(f'cache/{epoch}/{uuid}.pth')
    cache_nl, vehicle_type, vehicle_color = cache_nl['nls'], cache_nl['type'], cache_nl['color']
    # nls = []
    # for nl in query_nl:
    #     nl = torch.tensor(dataset.nl.sentence_to_index(nl, is_train=False)).cuda()
    #     nls.append(nl.unsqueeze(0).transpose(1, 0))
    uuids_per_nl = []
    prob_per_nl = []
    for idx, (id, frames, boxes, paths, rois, labels) in enumerate(tqdm(loader)):
        with torch.no_grad():
            boxes = boxes.squeeze(0).numpy()
            rois = rois.squeeze(0).numpy()
            # print(rois)
            frames = frames.squeeze(0)
            # print(frames.shape)
            # b = frames.shape[0]
            labels = labels.squeeze(0)
            # text = query_nl[0]
            
            # if idx in mem:
            #     cache = mem[idx]
            # else:
            
            # if num_of_vehicles == None:
            #     # version 3
            #     results = []
            #     caches = sorted(glob(f'cache/{epoch}/{idx}_*'), key = lambda x: int(x.split('_')[-1][:-4]))
            #     for cache_path in caches:
            #         c = torch.load(cache_path)

            #         output = model(cache_nl[0].expand(c.shape[0], -1, -1), c).sigmoid() +\
            #                 model(cache_nl[1].expand(c.shape[0], -1, -1), c).sigmoid() +\
            #                 model(cache_nl[2].expand(c.shape[0], -1, -1), c).sigmoid()
            #         results.append(output / 3)
            #     results = torch.cat(results, dim=0).cpu().detach().numpy()
                # cache = torch.load(f'cache/{epoch}/{idx}.pth')
                # b = cache.shape[0]
                # if b <= test_batch_size:
                #     cache_nl_ = cache_nl[0].expand(cache.shape[0], -1, -1)
                #     results = model(cache_nl_.cuda(), cache.cuda()).sigmoid().cpu().detach().numpy()
                # else:
                #     results = []
                #     for c in cache.split(test_batch_size):
                #         cache_nl_ = cache_nl[0].expand(c.shape[0], -1, -1)
                #         output = model(cache_nl_.cuda(), c.cuda()).sigmoid()
                #         results.append(output.cpu())
                #     results = torch.cat(results, dim=0).cpu().detach().numpy()
                # for batch_idx in range(cache.shape[0]):
                #     output = model(cache_nl[0], cache[batch_idx:batch_idx+1]).sigmoid()
                #     results.append(output.squeeze(0).cpu().detach().numpy())
            
            # else:
                # version 2
            cache = torch.load(f'cache/{epoch}/{idx}_0.pth')
            results = []
            cs = []
            vs = []
            nl1 = cache_nl[0]
            nl2 = cache_nl[1]
            nl3 = cache_nl[2]
            for frame, label in zip(cache.split(num_of_vehicles, dim=0), labels.split(num_of_vehicles, dim=0)):
                frame = frame.cuda()
                label = label.cuda()
                bs = frame.shape[0]
                # cache = cache[:num_of_vehicles]
                # print(cache.shape)
                if nl1.shape[0] != bs:
                    nl1 = cache_nl[0].expand(bs, -1, -1).cuda()
                    nl2 = cache_nl[1].expand(bs, -1, -1).cuda()
                    nl3 = cache_nl[2].expand(bs, -1, -1).cuda()
                # activation_aggregation, c_aggregation, v_aggregation = model(nl1, frame, label)

                # am, c, v = model(
                #     torch.cat([nl1, nl2, nl3], dim=0), 
                # frame.repeat(3, 1, 1, 1), 
                # label.repeat(3, 1, 1, 1))
                # print(frame.expand(bs*3, -1, -1, -1).shape)
                # am1, am2, am3 = am.split(bs)
                # c1, c2, c3 = c.split(bs)
                # v1, v2, v3 = v.split(bs)
                am1, c1, v1 = model(nl1, frame, label)
                am2, c2, v2 = model(nl2, frame, label)
                am3, c3, v3 = model(nl3, frame, label)
                activation_aggregation = (am1 + am2 + am3) / 3
                c_aggregation = (c1 + c2 + c3) / 3
                v_aggregation = (v1 + v2 + v3) / 3
                # activation_aggregation = model(nl1, frame) +\
                #     model(nl2, frame) +\
                #     model(nl3, frame)
                # activation_aggregation = activation_aggregation / 3
                results.append(activation_aggregation)
                cs.append(c_aggregation)
                vs.append(v_aggregation)

            results = torch.cat(results, dim=0).cpu().numpy()
            cs = torch.cat(cs, dim=0).mean(dim=0).cpu().numpy()
            vs = torch.cat(vs, dim=0).mean(dim=0).cpu().numpy()
            
            # version 1
            # cache = torch.load(f'cache/{epoch}/{idx}.pth')
            # results = []
            # for batch_idx in range(cache.shape[0]):
            #     output = model(cache_nl[0], cache[batch_idx:batch_idx+1]).sigmoid()
            #     results.append(output.squeeze(0).cpu().detach().numpy())

            prob = compute_probability_of_activations(results, rois, scene_threshold)
            # print(idx, ': ', prob)
            # if not os.path.exists('results/' + query_nl[0]):
            #     os.mkdir('results/' + query_nl[0])
            # if prob >= total_threshold:
            #     cs = np.argmax(cs)
            #     cs = CityFlowNLDataset.colors[cs]
            #     vs = np.argmax(vs)
            #     vs = CityFlowNLDataset.vehicle_type[vs]
            #     print(f'color: {cs}, type: {vs}')
            #     save_img(np.squeeze(results[0], axis=0) * 255, cv2.imread(paths[0][0]), boxes[0], f"results/{query_nl[0]}/{idx}_{prob}.png")
            
            # for submission
            # check color and type
            if vehicle_type != -1 and np.argmax(vs) != vehicle_type:
                prob = 0.
            if vehicle_color != -1 and np.argmax(cs) != vehicle_color:
                prob = 0.
            uuids_per_nl.append(id[0])
            prob_per_nl.append(prob)
    
    uuids_per_nl = np.array(uuids_per_nl)
    # print(uuids_per_nl.shape)
    prob_per_nl = np.array(prob_per_nl)
    prob_per_nl_arg = (-prob_per_nl).argsort(axis=0)
    sorted_uuids_per_nl = uuids_per_nl[prob_per_nl_arg]
    # print(prob_per_nl[prob_per_nl_arg])
    final_results[uuid] = sorted_uuids_per_nl.tolist()
    print(len(final_results.keys()))
    with open(f'results/submit_{epoch}.json', 'w') as fp:
        json.dump(final_results, fp)