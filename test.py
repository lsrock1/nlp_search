from dataset import CityFlowNLDataset, CityFlowNLInferenceDataset, query
from configs import get_default_config
from model import MyModel
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss

from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import MultiStepLR
import cv2
import numpy as np
import os


cfg = get_default_config()
dataset = CityFlowNLInferenceDataset(cfg, build_transforms(cfg))
# print(len(dataset.nl))
model = MyModel(cfg, len(dataset.nl)).cuda()
# optimizer = torch.optim.Adam(
#         params=model.parameters(),
#         lr=cfg.TRAIN.LR.BASE_LR)
# lr_scheduler = MultiStepLR(optimizer,
#                           milestones=(30, 60),
#                           gamma=cfg.TRAIN.LR.WEIGHT_DECAY)

loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=cfg.TRAIN.NUM_WORKERS)
uuids, nls = query(cfg)
epoch = 2
threshold = 0.5
model.load_state_dict(torch.load(f'save/{epoch}.pth'))
model.eval()
for uuid, query_nl in zip(uuids, nls):
    for idx, (id, frames, boxes, paths, rois) in enumerate(loader):
        with torch.no_grad():
            frames = frames.squeeze(0)
            # print(frames.shape)
            b = frames.shape[0]
            text = query_nl[0]
            # print(nl)
            nl = torch.tensor(dataset.nl.sentence_to_index(query_nl[0])).cuda()
            nl = nl.unsqueeze(0)#.expand(b, -1)
            nl = nl.transpose(1, 0)
            frames = frames.cuda()
            results = []
            for batch_idx in range(b):
                output = model(nl, frames[batch_idx:batch_idx+1]).sigmoid()
                results.append(output.squeeze(0))
            for i, o in enumerate(results):
                # print(paths)
                roi = [b.item() for b in rois[i]]
                img = cv2.imread(paths[i][0])
                o = (o > 0.5) * o
                # calculate activation ratio in roi
                activation_ratio = o[:, roi[1]:roi[3], roi[0]:roi[2]].sum() / ((roi[3] - roi[1]) * (roi[2] - roi[0]))
                if activation_ratio.item() < threshold:
                    continue
                o = o.squeeze(0).cpu().detach().numpy() * 255
                heatmap = cv2.applyColorMap(np.uint8(o), cv2.COLORMAP_JET)
                # draw box to heatmap
                heatmap = np.float32(heatmap) / 255
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                # img = cv2.resize(img, (384, 384))#(heatmap.shape[1], heatmap.shape[0]))

                cam = heatmap + np.float32(img) / 255
                cam = cam / np.max(cam)

                # h_ratio = img.shape[0] / o.shape[0]
                # w_ratio = img.shape[1] / o.shape[1]
                # xyxy
                box = [b.item() for b in boxes[i]]
                box = tuple([int(b) for b in box])
                cv2.rectangle(cam, box[:2], box[2:], (255,255,0), 2)
                cam = cam * 255
                cam = np.concatenate([cam, img], axis=1)

                if not os.path.exists('results/' + query_nl[0]):
                    os.mkdir('results/' + query_nl[0])
                cv2.imwrite(f"results/{query_nl[0]}/{idx}_{i}.png", cam.astype(np.uint8))
                print('saved: ', f"results/{query_nl[0]}/{idx}_{i}.png")
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