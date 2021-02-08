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
epoch = 5
model.load_state_dict(torch.load(f'my_{epoch}.pth'))
model.eval()
for uuid, query_nl in zip(uuids, nls):
    for idx, (id, frames, boxes, paths) in enumerate(loader):
        with torch.no_grad():
            frames = frames.squeeze(0)
            # print(frames.shape)
            b = frames.shape[0]
            text = query_nl[0]
            # print(nl)
            nl = torch.tensor(dataset.nl.sentence_to_index(query_nl[0])).cuda()
            nl = nl.unsqueeze(0).expand(b, -1)
            nl = nl.transpose(1, 0)
            frames = frames.cuda()
            output = model(nl, frames).sigmoid()
            for i, o in enumerate(output):
                # print(paths)
                img = cv2.imread(paths[i][0])
                o = o.squeeze(0).cpu().detach().numpy() * 255
                heatmap = cv2.applyColorMap(np.uint8(o), cv2.COLORMAP_JET)
                heatmap = np.float32(heatmap) / 255
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                cam = heatmap + np.float32(img) / 255
                cam = cam / np.max(cam)
                # print(o.shape)
                # print(paths[i])
                if not os.path.exists('results/' + query_nl[0]):
                    os.mkdir('results/' + query_nl[0])
                cv2.imwrite(f"results/{query_nl[0]}/{idx}_{i}.png", (cam * 255).astype(np.uint8))

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