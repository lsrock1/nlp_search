from dataset import CityFlowNLDataset
from configs import get_default_config
from model import MyModel
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss, sampling_loss

from torch.utils.data import DataLoader
import torch
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F


resume_epoch = 0

cfg = get_default_config()
dataset = CityFlowNLDataset(cfg, build_transforms(cfg))

model = MyModel(cfg, len(dataset.nl)).cuda()
optimizer = torch.optim.Adam(
        params=model.parameters(),
        lr=cfg.TRAIN.LR.BASE_LR, weight_decay=0.00003)
lr_scheduler = MultiStepLR(optimizer,
                          milestones=(30, 50, 70),
                          gamma=cfg.TRAIN.LR.WEIGHT_DECAY)

if resume_epoch > 0:
    model.load_state_dict(torch.load(f'save/{resume_epoch}.pth'))
    optimizer.load_state_dict(torch.load(f'save/{resume_epoch}_optim.pth'))
    lr_scheduler.last_epoch = resume_epoch
    lr_scheduler.step()
    print(f'resume from {resume_epoch} pth file, starting {resume_epoch+1} epoch')

loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)

for epoch in range(resume_epoch+1, cfg.TRAIN.EPOCH):
    losses = 0.
    precs = 0.
    for idx, (nl, frame, label) in enumerate(loader):
        # print(nl.shape)
        # print(global_img.shape)
        # print(local_img.shape)
        nl = nl.cuda()
        label = label.cuda()
        # global_img, local_img = global_img.cuda(), local_img.cuda()
        nl = nl.transpose(1, 0)
        frame = frame.cuda()
        # local_img = local_img.reshape(-1, 3, cfg.DATA.LOCAL_CROP_SIZE[0], cfg.DATA.LOCAL_CROP_SIZE[1])
        # global_img = global_img.reshape(-1, 3, cfg.DATA.GLOBAL_SIZE[0], cfg.DATA.GLOBAL_SIZE[1])
        output = model(nl, frame)
        # label_nl = torch.arange(nl.shape[0]).cuda()
        # label_img = label_nl.unsqueeze(1).expand(-1, cfg.DATA.NUM_IMG).flatten(start_dim=0).cuda()
        # loss, prec = triplet(nl, img_ft, label_nl, label_img)
        # print(label.sum(), ' ', (label == 0).sum())
        pred = (output.sigmoid() > 0.5)
        # print((pred == label).sum())
        pred = (pred == label) 
        recall = (pred * label).sum() / label.sum()

        accu = pred.sum().item() / pred.numel()
        # print(pred.sum(), ' ', label.sum())
        loss = sampling_loss(output, label)
        # loss = F.binary_cross_entropy_with_logits(output, label)
        # loss = sigmoid_focal_loss(output, label, reduction='mean')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses += loss.item()
        # precs += recall.item()
        lr = optimizer.param_groups[0]['lr']
        print(f'epoch: {epoch}, lr: {lr}, step: {idx}/{len(loader)}, loss: {losses / (idx + 1)}, recall: {recall.item()}, accuracy: {accu}')
    lr_scheduler.step()
    torch.save(model.state_dict(), f'save/{epoch}.pth')
    torch.save(optimizer.state_dict(), f'save/{epoch}_optim.pth')