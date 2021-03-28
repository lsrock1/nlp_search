from dataset import CityFlowNLDataset, CityFlowNLInferenceDataset, query
from configs import get_default_config
from model import MyModel
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss
from utils import compute_probability_of_activations, save_img

import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
from torch import nn
import torch.nn.functional as F
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


def main():
    epoch = 9
    test_batch_size = 64
    scene_threshold = 0.
    total_threshold = 0.2
    num_of_vehicles = 64

    cfg = get_default_config()
    dataset = CityFlowNLInferenceDataset(cfg, build_transforms(cfg), num_of_vehicles)

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    uuids, nls = query(cfg)

    if os.path.exists('results'):
        shutil.rmtree('results')
    os.mkdir('results')
    extract_cache_features(cfg, epoch, loader, dataset, test_batch_size, uuids, nls)
    cfg.num_gpu = torch.cuda.device_count()
    cfg.resume_epoch = 0
    mp.spawn(test, args=(cfg, loader, dataset, epoch, uuids, nls, scene_threshold, total_threshold),
                 nprocs=cfg.num_gpu, join=True)


def extract_cache_features(cfg, epoch, loader, dataset, test_batch_size, uuids, nls):
    # extract img fts first to save time
    if not os.path.exists('cache'):
        # shutil.rmtree('cache')
        os.mkdir('cache')

    if not os.path.exists(f'cache/{epoch}'):
        os.mkdir(f'cache/{epoch}')
        model = MyModel(cfg, len(dataset.nl), dataset.nl.word_to_idx['<PAD>'], nn.BatchNorm2d, num_colors=len(CityFlowNLDataset.colors), num_types=len(CityFlowNLDataset.vehicle_type) - 2).cuda()
        saved_dict = torch.load(f'save/{epoch}.pth')

        n = {}
        for k, v in saved_dict.items():
            n[k.replace('module.', '')] = v

        model.load_state_dict(n, False)
        model.eval()
        with torch.no_grad():
            for idx, (id, frames, _, _, _, labels) in enumerate(tqdm(loader)):
                frames = frames.squeeze(0).cuda()
                labels = labels.squeeze(0).cuda()
                # b = frames.shape[0]
                # cache = []

                # version 3
                # if b <= test_batch_size:
                #     cache = model.cnn(frames)
                #     torch.save(cache, f'cache/{epoch}/{idx}_0.pth')
                # else:
                #     cache = []
                for i, (f, l) in enumerate(zip(frames.split(test_batch_size), labels.split(test_batch_size))):
                    cache = model(None, f, l)
                    img_ft = cache[0]
                    color = F.softmax(cache[1].mean(dim=0), dim=0).cpu()#.numpy()
                    typ = F.softmax(cache[2].mean(dim=0), dim=0).cpu()#.numpy()
                    # color = np.argmax(color)
                    # typ = np.argmax(typ)

                    cache = [img_ft, color, typ]
                    torch.save(cache, f'cache/{epoch}/{idx}_{i}.pth')
                    break

            print('saving language features..')
            for uuid, query_nl in zip(uuids, nls):
                nls_list = []
                query_nl, vehicle_type = CityFlowNLDataset.type_replacer(query_nl)
                query_nl, vehicle_color = CityFlowNLDataset.color_replacer(query_nl)
                # max_len = max([len(dataset.nl.do_clean(nl)) for nl in query_nl])
                for nl in query_nl:
                    nl = torch.tensor(dataset.nl.sentence_to_index(nl, is_train=False)).cuda()
                    # nls.append(nl.unsqueeze(0).transpose(1, 0))
                    nl = nl.unsqueeze(0)#.transpose(1, 0)
                    # bs, len, dim
                    nl = model.rnn(nl)
                    nls_list.append(nl)
                saved_nls = {
                    'nls': nls_list,
                    'type': vehicle_type, 'color': vehicle_color
                }
                torch.save(saved_nls, f'cache/{epoch}/{uuid}.pth')
        # model = model.cpu()
        del model, saved_dict, n, nls_list, img_ft
        torch.cuda.empty_cache()


def test(rank, cfg, loader, dataset, epoch, uuids, nls, scene_threshold, total_threshold):
    dataset.load_frame = False
    dist.init_process_group(backend="nccl", rank=rank,
                            world_size=cfg.num_gpu,
                            init_method="env://")
    torch.cuda.set_device(rank)
    cudnn.benchmark = True
    model = MyModel(cfg, len(dataset.nl), dataset.nl.word_to_idx['<PAD>'], norm_layer=nn.BatchNorm2d, num_colors=len(CityFlowNLDataset.colors), num_types=len(CityFlowNLDataset.vehicle_type) - 2).cuda()
    model = DistributedDataParallel(model, device_ids=[rank],
                                    output_device=rank,
                                    broadcast_buffers=cfg.num_gpu > 1, find_unused_parameters=False)
    saved_dict = torch.load(f'save/{epoch}.pth', map_location=torch.device(f'cuda:{rank}'))
    model.load_state_dict(saved_dict, True)
    model.eval()
    final_results = {}

    a = len(nls) // cfg.num_gpu
    start = a * rank
    end = None if (rank + 1) == cfg.num_gpu else a * (rank + 1)
    end_str = 'end' if end == None else end
    print(f'process number: {rank}, {start}:{end_str}')
    for nlidx, (uuid, query_nl) in enumerate(zip(uuids[start:end], nls[start:end])):
        nlidx = nlidx + start
        print(f'{nlidx} / {len(nls)}')
        cache_nl = torch.load(f'cache/{epoch}/{uuid}.pth', map_location=torch.device(f'cuda:{rank}'))
        cache_nl, vehicle_type, vehicle_color = cache_nl['nls'], cache_nl['type'], cache_nl['color']
        # nls = []
        # for nl in query_nl:
        #     nl = torch.tensor(dataset.nl.sentence_to_index(nl, is_train=False)).cuda()
        #     nls.append(nl.unsqueeze(0).transpose(1, 0))
        uuids_per_nl = []
        prob_per_nl = []
        for idx, (id, frames, boxes, paths, rois, labels) in enumerate(loader):
            # print(f'{nlidx}_{idx}')
            with torch.no_grad():
                boxes = boxes.squeeze(0).numpy()
                rois = rois.squeeze(0).numpy()
                # print(rois)
                frames = frames.squeeze(0)
                # print(frames.shape)
                # b = frames.shape[0]
                labels = labels.squeeze(0)
                labels = labels.cuda()

                cache = torch.load(f'cache/{epoch}/{idx}_0.pth', map_location=torch.device(f'cuda:{rank}'))
                # print(cache)
                frame, cs, vs = cache
                # if vehicle_type != -1 and vs != vehicle_type:
                #     continue
                # if vehicle_color != -1 and cs != vehicle_color:
                #     continue
                # print(frame.device)
                # results = []
                
                nl1 = cache_nl[0]
                nl2 = cache_nl[1]
                nl3 = cache_nl[2]
                
                bs = frame.shape[0]
                # cache = cache[:num_of_vehicles]
                # print(cache.shape)
                if nl1.shape[0] != bs:
                    nl1 = cache_nl[0].expand(bs, -1, -1).cuda()
                    nl2 = cache_nl[1].expand(bs, -1, -1).cuda()
                    nl3 = cache_nl[2].expand(bs, -1, -1).cuda()

                am1 = model(nl1, frame, labels)
                am2 = model(nl2, frame, labels)
                am3 = model(nl3, frame, labels)
                # am1, c1, v1 = model(nl1, frame, labels)
                # am2, c2, v2 = model(nl2, frame, labels)
                # am3, c3, v3 = model(nl3, frame, labels)
                activation_aggregation = (am1 + am2 + am3) / 3
                # c_aggregation = (c1 + c2 + c3) / 3
                # v_aggregation = (v1 + v2 + v3) / 3
                # activation_aggregation = model(nl1, frame) +\
                #     model(nl2, frame) +\
                #     model(nl3, frame)
                # activation_aggregation = activation_aggregation / 3
                # results.append(activation_aggregation)
                # cs.append(c_aggregation)
                # vs.append(v_aggregation)

                results = activation_aggregation.cpu().numpy()
                # cs = c_aggregation.mean(dim=0).cpu().numpy()
                # vs = v_aggregation.mean(dim=0).cpu().numpy()
                
                # version 1
                # cache = torch.load(f'cache/{epoch}/{idx}.pth')
                # results = []
                # for batch_idx in range(cache.shape[0]):
                #     output = model(cache_nl[0], cache[batch_idx:batch_idx+1]).sigmoid()
                #     results.append(output.squeeze(0).cpu().detach().numpy())

                prob = compute_probability_of_activations(results, rois, scene_threshold)
                
                # if vehicle_type != -1 and np.argmax(vs) != vehicle_type:
                #     prob = 0.
                # if vehicle_color != -1 and np.argmax(cs) != vehicle_color:
                #     prob = 0.

                ###### visualization
                # if not os.path.exists('results/' + query_nl[0]):
                #     os.mkdir('results/' + query_nl[0])
                
                # # cs = np.argmax(cs)
                # cs = cs.item()
                # vs = vs.item()
                # cs = CityFlowNLDataset.colors[cs]
                # # vs = np.argmax(vs)
                # vs = CityFlowNLDataset.vehicle_type[vs]
                # if prob > total_threshold:
                    
                #     print(f'color: {cs}, type: {vs}')
                #     save_img(np.squeeze(results[0], axis=0) * 255, cv2.imread(paths[0][0]), boxes[0], f"results/{query_nl[0]}/{idx}_{prob}.png")
                
                ###### end visualization


                # for submission
                uuids_per_nl.append(id[0])
                prob_per_nl.append(prob)
        final_results['uuids_order'] = uuids_per_nl
        final_results[uuid] = prob_per_nl

        # uuids_per_nl = np.array(uuids_per_nl)
        # # print(uuids_per_nl.shape)
        # prob_per_nl = np.array(prob_per_nl)
        # prob_per_nl_arg = (-prob_per_nl).argsort(axis=0)
        # sorted_uuids_per_nl = uuids_per_nl[prob_per_nl_arg]
        # # print(prob_per_nl[prob_per_nl_arg])
        # final_results[uuid] = sorted_uuids_per_nl.tolist()
        # print(len(final_results.keys()))

        with open(f'results/submit_{epoch}_{start}_{end_str}.json', 'w') as fp:
            json.dump(final_results, fp)


if __name__ == '__main__':
    main()