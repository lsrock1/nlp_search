import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
import math


class PositionalEncoding2D(nn.Module):

    def __init__(self, d_model, height_max, width_max, dropout=0.1):
        super(PositionalEncoding2D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(d_model, height_max, width_max)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width_max).unsqueeze(1)
        pos_h = torch.arange(0., height_max).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height_max, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height_max, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width_max)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width_max)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[2], :x.shape[3]]
        return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RNN(nn.Module):
    def __init__(self, cfg, num_words, padding_idx):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = padding_idx
        self.pos_encoder = PositionalEncoding(cfg.MODEL.RNN.HIDDEN, max_len=30)
        self.embedding = nn.Embedding(num_words, cfg.MODEL.RNN.HIDDEN)
        self.rnn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=cfg.MODEL.RNN.HIDDEN, nhead=8),
            num_layers=cfg.MODEL.RNN.LAYERS
        )
        self.src_mask = None
        self.hidden_size = cfg.MODEL.RNN.HIDDEN
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def make_pad_mask(self, idx):
        # len, bs to bs, len
        return (idx == self.padding_idx).transpose(1, 0)

    def forward(self, x):
        if self.src_mask is None or self.src_mask.size(0) != len(x):
            device = x.device
            mask = self._generate_square_subsequent_mask(len(x)).to(device)
            self.src_mask = mask
        mask = self.make_pad_mask(x)
        # print(mask)
        x = self.embedding(x) * math.sqrt(self.hidden_size)
        x = self.pos_encoder(x)
        x = self.rnn(x, self.src_mask, src_key_padding_mask=mask)
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
    def __init__(self, cfg, norm_layer):
        super().__init__()
        self.cfg = cfg
        # self.global_embedding = models.vgg16(pretrained=True).features
        self.global_embedding = models.resnet50(pretrained=True, norm_layer=norm_layer)
        self.global_embedding.layer4[0].conv2.stride = 1
        self.global_embedding.layer4[0].downsample[0].stride = 1
        self.global_embedding = nn.Sequential(
            self.global_embedding.conv1,
            self.global_embedding.bn1,
            self.global_embedding.relu,
            self.global_embedding.maxpool,
            self.global_embedding.layer1,
            self.global_embedding.layer2,
            self.global_embedding.layer3,
            self.global_embedding.layer4
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
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, t, c = x.size()
        y = x.mean(dim=1)
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(y)
        return y


class MyModel(nn.Module):
    def __init__(self, cfg, num_words, padding_idx, norm_layer, num_colors, num_types):
        super().__init__()
        if norm_layer == None:
            norm_layer = nn.BatchNorm2d
        self.rnn = RNN(cfg, num_words, padding_idx)
        self.cnn = CNN(cfg, norm_layer)
        self.a = nn.Linear(2048, 2048)
        self.b = nn.Conv2d(2048, 2048, 1)
        self.c = nn.Linear(2048, 2048)
        self.d = nn.Conv2d(2048, 2048, 1)

        self.se = SELayer(2048)
        self.out = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), norm_layer(1024), nn.ReLU(True),
            nn.Conv2d(1024, 512, 3, padding=1), norm_layer(512), nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1), norm_layer(256), nn.ReLU(True),
            nn.Conv2d(256, 1, 3, padding=1))
        
        if norm_layer == nn.BatchNorm2d:
            norm_layer = nn.BatchNorm1d

        self.color = nn.Sequential(
            norm_layer(2048), nn.Linear(2048, num_colors)
        )
        self.types = nn.Sequential(
            norm_layer(2048), nn.Linear(2048, num_types)
        )

        self.pos = PositionalEncoding2D(2048, 24, 24)
        self.attn = nn.ModuleList([
            nn.Conv2d(2048, 512, 1), nn.Conv2d(2048, 2048, 1), nn.Conv2d(2048, 2048, 1)
        ])

    def forward(self, nl, global_img, activation_map=None):
        if self.training:
            nl = self.rnn(nl)
            img_ft = self.cnn(global_img)
        else:
            img_ft = global_img

        bs, length, emb = nl.shape
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
        
        # ===== self attention =====
        img_ft = self.pos(img_ft)
        img_key = self.attn[0](img_ft).reshape(bs, 512, -1)
        # bs, hw, hw
        img_key = img_key.permute(0, 2, 1).contiguous().bmm(img_key)
        img_key = F.softmax(img_key, dim=-1)

        # bs, c, hw
        img_value = self.attn[1](img_ft).reshape(bs, 2048, -1)
        # bs, c, h, w
        img_value = img_value.bmm(img_key.permute(0, 2, 1)).reshape(bs, c, h, w)
        img_ft = img_ft + img_value
        img_ft = self.attn[2](img_ft)
        # ===== end self attention =====

        nl = nl.reshape(nl.shape[0], -1, 1, 1)
        last = img_ft + nl
        # nl = nl.expand(-1, -1, img_ft.shape[2], img_ft.shape[3])
        # last = torch.cat([img_ft, nl], dim=1)
        if not self.training:
            return self.out(last)#.sigmoid()
        else:
            pred_map = self.out(last)
            vectors = (last * activation_map).sum(dim=(2, 3)) / activation_map.sum(dim=(2, 3))
            color = self.color(vectors)
            types = self.types(vectors)
            return pred_map, color, types
