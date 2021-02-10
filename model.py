import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, cfg, num_words):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(num_words, cfg.MODEL.RNN.HIDDEN)
        self.rnn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.MODEL.RNN.HIDDEN, nhead=8),
            num_layers=cfg.MODEL.RNN.LAYERS
        )
        # nn.GRU(
        #     input_size=cfg.MODEL.RNN.EMBEDDING,
        #     hidden_size=cfg.MODEL.RNN.HIDDEN,
        #     num_layers=cfg.MODEL.RNN.LAYERS,
        #     bidirectional=True,
        #     batch_first=True
        # )
        # self.linears = nn.ModuleList([
        #     nn.Linear(cfg.MODEL.RNN.HIDDEN, cfg.MODEL.RNN.HIDDEN),
        #     nn.Sequential(
        #         nn.Linear(cfg.MODEL.RNN.HIDDEN, cfg.MODEL.RNN.HIDDEN//2, bias=False),
        #         nn.BatchNorm1d(cfg.MODEL.RNN.HIDDEN//2), nn.ReLU(True),
        #         nn.Linear(cfg.MODEL.RNN.HIDDEN//2, 1)),
        #     nn.Sequential(
        #         nn.Linear(cfg.MODEL.RNN.HIDDEN, cfg.MODEL.RNN.HIDDEN//2, bias=False),
        #         nn.BatchNorm1d(cfg.MODEL.RNN.HIDDEN//2), nn.ReLU(True),
        #         nn.Linear(cfg.MODEL.RNN.HIDDEN//2, 1))
        # ])

    def forward(self, x):
        x = self.embedding(x)
        x = self.rnn(x)
        length, bs, emb = x.shape
        return x.permute(1, 0, 2)
        # global_ratio = self.linears[1](x.reshape(-1, emb)).reshape(length, bs, 1)
        # global_ratio = F.softmax(global_ratio, dim=0)
  
        # local_ratio = self.linears[2](x.reshape(-1, emb)).reshape(length, bs, 1)
        # local_ratio = F.softmax(local_ratio, dim=0)
        # x = self.linears[0](x.reshape(-1, emb)).reshape(length, bs, -1)
        # global_feature = (global_ratio * x).sum(dim=0)
        # local_vector = (local_ratio * x).sum(dim=0)
        # return torch.cat([global_feature, local_vector], dim=-1)


class CNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.global_embedding = models.vgg16(pretrained=True).features
        self.global_embedding = models.resnet50(pretrained=True)
        self.global_embedding = nn.Sequential(
            self.global_embedding.conv1,
            self.global_embedding.bn1,
            self.global_embedding.relu,
            self.global_embedding.maxpool,
            self.global_embedding.layer1,
            self.global_embedding.layer2,
            self.global_embedding.layer3,
            # self.global_embedding.layer4
        )
        #
        # self.local_embedding = models.resnet50(pretrained=True)
        # self.local_embedding = nn.Sequential(
        #     self.local_embedding.conv1,
        #     self.local_embedding.bn1,
        #     self.local_embedding.relu,
        #     self.local_embedding.maxpool,
        #     self.local_embedding.layer1,
        #     self.local_embedding.layer2,
        #     self.local_embedding.layer3,
        #     self.local_embedding.layer4, nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1)
        # )
        # self.local_embedding = nn.Sequential(
        #    models.vgg16(pretrained=True).features, nn.AdaptiveAvgPool2d(1), nn.Flatten(start_dim=1))

        # self.linears = nn.ModuleList([
        #     nn.Conv2d(512, cfg.MODEL.CNN.ATTN_CHANNEL, 1), nn.Linear(512, cfg.MODEL.CNN.ATTN_CHANNEL)])
        # self.out_linear = nn.ModuleList([
        #     nn.Conv2d(512, cfg.MODEL.CNN.ATTN_CHANNEL, 1), nn.Linear(512, cfg.MODEL.CNN.ATTN_CHANNEL)])

    def forward(self, global_img):
        return self.global_embedding(global_img)
        # global_feature = self.global_embedding(global_img)
        # local_vector = self.local_embedding(local_img)

        # weights = self.linears[0](global_feature).flatten(start_dim=2).permute(0, 2, 1).bmm(
        #     self.linears[1](local_vector).unsqueeze(-1)).permute(0, 2, 1)
        # weights = F.conv2d(self.linears[0](global_feature), local_vector).flatten(start_dim=2)
        # weights = F.softmax(weights, dim=-1)
        # print(weights.shape)
        # print(global_feature.shape)
        # global_feature = self.out_linear[0](global_feature).flatten(start_dim=2) * weights
        # global_feature = global_feature.sum(dim=-1)

        # return torch.cat([global_feature, self.out_linear[1](local_vector.flatten(start_dim=1))], dim=1)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c = x.size()
        y = x.mean(dim=1)
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y


class MyModel(nn.Module):
    def __init__(self, cfg, num_words):
        super().__init__()
        self.rnn = RNN(cfg, num_words)
        self.cnn = CNN(cfg)
        self.a = nn.Linear(1024, 1024)
        self.b = nn.Conv2d(1024, 1024, 1)
        self.c = nn.Linear(1024, 1024)
        self.d = nn.Conv2d(1024, 1024, 1)

        self.se = SELayer(1024)
        self.out = nn.Sequential(
            nn.BatchNorm2d(2048), nn.ReLU(True),
            nn.Conv2d(2048, 1024, 3, padding=1), nn.BatchNorm2d(1024), nn.ReLU(True),
            nn.Conv2d(1024, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 1, 3, padding=1))

    def forward(self, nl, global_img):
        if self.training:
            nl = self.rnn(nl)
            img_ft = self.cnn(global_img)
        else:
            img_ft = global_img

        img_ft_b = self.b(img_ft)
        bs, c, h, w = img_ft_b.shape
        # bs, t, hw
        relation = torch.bmm(self.a(nl), img_ft_b.reshape(bs, c, -1))
        weights = F.softmax(relation, dim=1)
        weighted_img_ft = torch.bmm(self.c(nl).permute(0, 2, 1), weights)
        img_ft = weighted_img_ft.reshape(bs, c, h, w) + img_ft
        
        weights = F.softmax(relation, dim=2)
        weighted_nl_ft = torch.bmm(weights, self.d(img_ft).reshape(bs, c, -1).permute(0, 2, 1))
        nl = weighted_nl_ft + nl

        se = self.se(nl)

        nl = nl * se.unsqueeze(dim=1)
        nl = nl.mean(dim=1).unsqueeze(-1).unsqueeze(-1)
        img_ft = img_ft * se.unsqueeze(dim=-1).unsqueeze(dim=-1)
        nl = nl.reshape(nl.shape[0], -1, 1, 1)
        nl = nl.expand(-1, -1, img_ft.shape[2], img_ft.shape[3])
        last = torch.cat([img_ft, nl], dim=1)
        return self.out(last)#.sigmoid()
