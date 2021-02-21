from dataset import CityFlowNLDataset, CityFlowNLInferenceDataset, query
from configs import get_default_config
from model import MyFilm
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


epoch = 19
test_batch_size = 64
scene_threshold = 0.8
total_threshold = 0.8
num_of_vehicles = 64

cfg = get_default_config()
dataset = CityFlowNLInferenceDataset(cfg, build_transforms(cfg), num_of_vehicles)
model = MyFilm(cfg, len(dataset.nl), dataset.nl.word_to_idx['<PAD>'],num_colors=len(CityFlowNLDataset.colors), num_types=len(CityFlowNLDataset.vehicle_type) - 2).cuda()

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

cache_img_ft = {}
# extract img fts first to save time
if not os.path.exists('cache'):
    # shutil.rmtree('cache')
    os.mkdir('cache')

if not os.path.exists(f'cache/{epoch}'):
    os.mkdir(f'cache/{epoch}')

    with torch.no_grad():

        print('saving language features..')
        for uuid, query_nl in zip(uuids, nls):
            nls_list = []
            query_nl, vehicle_type = CityFlowNLDataset.type_replacer(query_nl)
            query_nl, vehicle_color = CityFlowNLDataset.color_replacer(query_nl)
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

# dataset.load_frame = False

mem = {}
for nlidx, (uuid, query_nl) in enumerate(zip(uuids, nls)):
    print(f'{nlidx} / {len(nls)}')
    cache_nl = torch.load(f'cache/{epoch}/{uuid}.pth')
    cache_nl, vehicle_type, vehicle_color = cache_nl['nls'], cache_nl['type'], cache_nl['color']
    # nls = []
    # for nl in query_nl:
    #     nl = torch.tensor(dataset.nl.sentence_to_index(nl, is_train=False)).cuda()
    #     nls.append(nl.unsqueeze(0).transpose(1, 0))
    for idx, (id, frames, boxes, paths, rois, labels) in enumerate(tqdm(loader)):
        with torch.no_grad():
            boxes = boxes.squeeze(0).numpy()
            rois = rois.squeeze(0).numpy()
            # print(rois)
            frames = frames.squeeze(0)
            # print(frames.shape)
            # b = frames.shape[0]
            labels = labels.squeeze(0)
            text = query_nl[0]
            
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
            # cache = torch.load(f'cache/{epoch}/{idx}_0.pth')
            results = []
            cs = []
            vs = []
            nl1 = cache_nl[0]
            nl2 = cache_nl[1]
            nl3 = cache_nl[2]
            for frame, label in zip(frames.split(num_of_vehicles, dim=0), labels.split(num_of_vehicles, dim=0)):
                frame = frame.cuda()
                label = label.cuda()
                # cache = cache[:num_of_vehicles]
                # print(cache.shape)
                if nl1.shape[0] != frame.shape[0]:
                    nl1 = cache_nl[0].expand(frame.shape[0], -1, -1).cuda()
                    nl2 = cache_nl[1].expand(frame.shape[0], -1, -1).cuda()
                    nl3 = cache_nl[2].expand(frame.shape[0], -1, -1).cuda()

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
                results.append(activation_aggregation.cpu().detach().numpy())
                cs.append(c_aggregation.cpu().detach().numpy())
                vs.append(v_aggregation.cpu().detach().numpy())
            results = np.concatenate(results, axis=0)
            cs = np.mean(np.concatenate(cs, axis=0), axis=0)
            vs = np.mean(np.concatenate(vs, axis=0), axis=0)
            
            # version 1
            # cache = torch.load(f'cache/{epoch}/{idx}.pth')
            # results = []
            # for batch_idx in range(cache.shape[0]):
            #     output = model(cache_nl[0], cache[batch_idx:batch_idx+1]).sigmoid()
            #     results.append(output.squeeze(0).cpu().detach().numpy())

            prob = compute_probability_of_activations(results, rois, scene_threshold)
            # print(idx, ': ', prob)
            if not os.path.exists('results/' + query_nl[0]):
                os.mkdir('results/' + query_nl[0])
            if prob >= total_threshold:
                cs = np.argmax(cs)
                cs = CityFlowNLDataset.colors[cs]
                vs = np.argmax(vs)
                vs = CityFlowNLDataset.vehicle_type[vs]
                print(f'color: {cs}, type: {vs}')
                save_img(np.squeeze(results[0], axis=0) * 255, cv2.imread(paths[0][0]), boxes[0], f"results/{query_nl[0]}/{idx}_{prob}.png")
            # for i, o in enumerate(results):
            #     # print(paths)
            #     roi = [b.item() for b in rois[i]]
            #     img = cv2.imread(paths[i][0])
            #     o = (o > 0.5) * o
            #     # calculate activation ratio in roi
            #     activation_ratio = o[:, roi[1]:roi[3], roi[0]:roi[2]].sum() / ((roi[3] - roi[1]) * (roi[2] - roi[0]))
            #     if activation_ratio.item() < threshold:
            #         continue
            #     o = o.squeeze(0).cpu().detach().numpy() * 255
            #     heatmap = cv2.applyColorMap(np.uint8(o), cv2.COLORMAP_JET)
            #     # draw box to heatmap
            #     heatmap = np.float32(heatmap) / 255
            #     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            #     # img = cv2.resize(img, (384, 384))#(heatmap.shape[1], heatmap.shape[0]))

            #     cam = heatmap + np.float32(img) / 255
            #     cam = cam / np.max(cam)

            #     # h_ratio = img.shape[0] / o.shape[0]
            #     # w_ratio = img.shape[1] / o.shape[1]
            #     # xyxy
            #     box = [b.item() for b in boxes[i]]
            #     box = tuple([int(b) for b in box])
            #     cv2.rectangle(cam, box[:2], box[2:], (255,255,0), 2)
            #     cam = cam * 255
            #     cam = np.concatenate([cam, img], axis=1)

            #     if not os.path.exists('results/' + query_nl[0]):
            #         os.mkdir('results/' + query_nl[0])
            #     cv2.imwrite(f"results/{query_nl[0]}/{idx}_{i}.png", cam.astype(np.uint8))
            #     print('saved: ', f"results/{query_nl[0]}/{idx}_{i}.png")
# for epoch in range(cfg.TRAIN.EPOCH):
#     losses = 0.
#     precs = 0.
#     for idx, (nl, frame, label) in enumerate(loader):
#         # print(nl.shape)
#         # print(global_img.shape)
#         # print(local_img.shape)
#         nl = nl.cuda()
#         # global_img, local_img = global_img.cuda(), local_img.cuda()
#         nl = nl.transpose(1, 0)
#         frame = frame.cuda()
#         # local_img = local_img.reshape(-1, 3, cfg.DATA.LOCAL_CROP_SIZE[0], cfg.DATA.LOCAL_CROP_SIZE[1])
#         # global_img = global_img.reshape(-1, 3, cfg.DATA.GLOBAL_SIZE[0], cfg.DATA.GLOBAL_SIZE[1])
#         output = model(nl, frame)
#         # label_nl = torch.arange(nl.shape[0]).cuda()
#         # label_img = label_nl.unsqueeze(1).expand(-1, cfg.DATA.NUM_IMG).flatten(start_dim=0).cuda()
#         # loss, prec = triplet(nl, img_ft, label_nl, label_img)
#         loss = sigmoid_focal_loss(output, label.cuda(), reduction='sum')
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         losses += loss.item()
#         # precs += prec
#         print(f'step: {idx}/{len(loader)}, loss: {losses / (idx + 1)}, prec: {precs / (idx + 1)}')
#     torch.save(model.state_dict(), f'my_{epoch}.pth')