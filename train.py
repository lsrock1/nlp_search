from dataset import CityFlowNLDataset
from configs import get_default_config
from model import MyModel
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss

from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import MultiStepLR


cfg = get_default_config()
dataset = CityFlowNLDataset(cfg, build_transforms(cfg))
print(len(dataset.nl))
model = MyModel(cfg, len(dataset.nl)).cuda()
optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.TRAIN.LR.BASE_LR, weight_decay=0.00003)
lr_scheduler = MultiStepLR(optimizer,
                          milestones=(30, 50, 70),
                          gamma=cfg.TRAIN.LR.WEIGHT_DECAY)

loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)

for epoch in range(cfg.TRAIN.EPOCH):
    losses = 0.
    precs = 0.
    for idx, (nl, frame, label) in enumerate(loader):
        # print(nl.shape)
        # print(global_img.shape)
        # print(local_img.shape)
        nl = nl.cuda()
        # global_img, local_img = global_img.cuda(), local_img.cuda()
        nl = nl.transpose(1, 0)
        frame = frame.cuda()
        # local_img = local_img.reshape(-1, 3, cfg.DATA.LOCAL_CROP_SIZE[0], cfg.DATA.LOCAL_CROP_SIZE[1])
        # global_img = global_img.reshape(-1, 3, cfg.DATA.GLOBAL_SIZE[0], cfg.DATA.GLOBAL_SIZE[1])
        output = model(nl, frame)
        # label_nl = torch.arange(nl.shape[0]).cuda()
        # label_img = label_nl.unsqueeze(1).expand(-1, cfg.DATA.NUM_IMG).flatten(start_dim=0).cuda()
        # loss, prec = triplet(nl, img_ft, label_nl, label_img)
        loss = sigmoid_focal_loss(output, label.cuda(), reduction='sum')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses += loss.item()
        # precs += prec
        print(f'epoch: {epoch} ,step: {idx}/{len(loader)}, loss: {losses / (idx + 1)}, prec: {precs / (idx + 1)}')
    torch.save(model.state_dict(), f'save/{epoch}.pth')