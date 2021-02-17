from dataset import CityFlowNLDataset
from configs import get_default_config
from model import MyModel, MyFilm
from transforms import build_transforms
from loss import TripletLoss, sigmoid_focal_loss, sampling_loss, reduce_sum, LabelSmoothingLoss
from scheduler import WarmupMultiStepLR

import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, DistributedSampler
import torch
from torch import nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import os


cfg = get_default_config()

def train_model_on_dataset(rank, cfg):
    dist_rank = rank
    # print(dist_rank)
    dist.init_process_group(backend="nccl", rank=dist_rank,
                            world_size=cfg.num_gpu,
                            init_method="env://")
    torch.cuda.set_device(rank)
    cudnn.benchmark = True
    dataset = CityFlowNLDataset(cfg, build_transforms(cfg))

    model = MyModel(cfg, len(dataset.nl), dataset.nl.word_to_idx['<PAD>'], norm_layer=nn.SyncBatchNorm, num_colors=len(dataset.colors), num_types=len(dataset.vehicle_type) - 2).cuda()
    model = DistributedDataParallel(model, device_ids=[rank],
                                    output_device=rank,
                                    broadcast_buffers=cfg.num_gpu > 1)
    optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.TRAIN.LR.BASE_LR, weight_decay=0.00003)
    lr_scheduler = WarmupMultiStepLR(optimizer,
                            milestones=cfg.TRAIN.STEPS,
                            gamma=cfg.TRAIN.LR.WEIGHT_DECAY,
                            warmup_factor=cfg.TRAIN.WARMUP_FACTOR,
                            warmup_iters=cfg.TRAIN.WARMUP_EPOCH)
    color_loss = LabelSmoothingLoss(len(dataset.colors), 0.1)
    vehicle_loss = LabelSmoothingLoss(len(dataset.vehicle_type) - 2, 0.1)
    if cfg.resume_epoch > 0:
        model.load_state_dict(torch.load(f'save/{cfg.resume_epoch}.pth'))
        optimizer.load_state_dict(torch.load(f'save/{cfg.resume_epoch}_optim.pth'))
        lr_scheduler.last_epoch = cfg.resume_epoch
        lr_scheduler.step()
        if rank == 0:
            print(f'resume from {cfg.resume_epoch} pth file, starting {cfg.resume_epoch+1} epoch')
        cfg.resume_epoch += 1

    # loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
    train_sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE //cfg.num_gpu,
                            num_workers=cfg.TRAIN.NUM_WORKERS // cfg.num_gpu,# shuffle=True,
                            sampler=train_sampler, pin_memory=True)
    for epoch in range(cfg.resume_epoch, cfg.TRAIN.EPOCH):
        losses = 0.
        losses_color = 0.
        losses_types = 0.
        precs = 0.
        train_sampler.set_epoch(epoch)
        for idx, (nl, frame, label, color_label, type_label) in enumerate(loader):
            # print(nl.shape)
            # print(global_img.shape)
            # print(local_img.shape)
            nl = nl.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            # global_img, local_img = global_img.cuda(), local_img.cuda()
            nl = nl.transpose(1, 0)
            frame = frame.cuda(non_blocking=True)
            # local_img = local_img.reshape(-1, 3, cfg.DATA.LOCAL_CROP_SIZE[0], cfg.DATA.LOCAL_CROP_SIZE[1])
            # global_img = global_img.reshape(-1, 3, cfg.DATA.GLOBAL_SIZE[0], cfg.DATA.GLOBAL_SIZE[1])
            output, color, types = model(nl, frame, label)
            # label_nl = torch.arange(nl.shape[0]).cuda()
            # label_img = label_nl.unsqueeze(1).expand(-1, cfg.DATA.NUM_IMG).flatten(start_dim=0).cuda()
            # loss, prec = triplet(nl, img_ft, label_nl, label_img)
            # print(label.sum(), ' ', (label == 0).sum())
            
            # print(pred.sum(), ' ', label.sum())
            # loss = sampling_loss(output, label)
            # loss = F.binary_cross_entropy_with_logits(output, label)
            total_num_pos = reduce_sum(label.new_tensor([label.sum()])).item()
            num_pos_avg_per_gpu = max(total_num_pos / float(cfg.num_gpu), 1.0)

            loss = sigmoid_focal_loss(output, label, reduction='sum') / num_pos_avg_per_gpu
            loss_color = color_loss(color, color_label.cuda())
            loss_type = vehicle_loss(types, type_label.cuda())
            loss_total = loss + loss_color + loss_type
            optimizer.zero_grad()
            loss_total.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            losses += loss.item()
            losses_color += loss_color.item()
            losses_types += loss_type.item()
            # precs += recall.item()
            
            if rank == 0:
                pred = (output.sigmoid() > 0.5)
                # print((pred == label).sum())
                pred = (pred == label) 
                recall = (pred * label).sum() / label.sum()

                accu = pred.sum().item() / pred.numel()
                lr = optimizer.param_groups[0]['lr']
                print(f'epoch: {epoch},', 
                f'lr: {lr}, step: {idx}/{len(loader)},',
                f'loss: {losses / (idx + 1):.4f},', 
                f'loss color: {losses_color / (idx + 1):.4f},',
                f'loss type: {losses_types / (idx + 1):.4f},',
                f'recall: {recall.item():.4f}, accuracy: {accu:.4f}')
        lr_scheduler.step()
        if rank == 0:
            if not os.path.exists('save'):
                os.mkdir('save')
            torch.save(model.state_dict(), f'save/{epoch}.pth')
            torch.save(optimizer.state_dict(), f'save/{epoch}_optim.pth')


if __name__ == '__main__':
    cfg.num_gpu = torch.cuda.device_count()
    cfg.resume_epoch = 0
    mp.spawn(train_model_on_dataset, args=(cfg,),
                 nprocs=cfg.num_gpu, join=True)