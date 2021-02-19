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
            nn.TransformerEncoderLayer(d_model=cfg.MODEL.RNN.HIDDEN, nhead=4),
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


class SELayer(nn.Module):
    def __init__(self, channel, out_channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, out_channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_channel // reduction, out_channel),
        )

    def forward(self, x):
        y = self.fc(x)
        return y


class FilmBlock(nn.Module):
    def __init__(self, block, rnn_dim, in_channel, channel, h, w, spatial=True, has_pos=True):
        super(FilmBlock, self).__init__()
        self.block = block
        self.channel = channel
        self.has_spatial = spatial
        self.has_pos = has_pos
        if spatial:
            self.embeddings = nn.ModuleList([
                nn.Conv2d(channel, channel, 1),
                nn.Linear(rnn_dim, channel),
                nn.Linear(rnn_dim, channel)
            ])
        self.se = SELayer(rnn_dim, channel * 2)
        if has_pos:
            self.pos = PositionalEncoding2D(in_channel, h, w)

    def forward(self, x, nl):
        if self.has_pos:
            x = self.pos(x)
        x = self.block(x)
        bs, c, h, w = x.shape
        # bs, c, hw
        x_a = self.embeddings[0](x).reshape(bs, c, -1)
        # bs, len, c
        nl_a = self.embeddings[1](nl)
        # bs, len, hw
        weights = nl_a.bmm(x_a)

        # bs, c, h, w
        # (bs, c, len) @ (bs, len, hw)
        x = self.embeddings[2](nl).permute(0, 2, 1).bmm(F.softmax(weights, dim=1)).reshape(bs, c, h, w) + x
        
        # bs, len, c
        # (bs, len, hw) @ (bs, hw, c)
        # nl = F.softmax(weights, dim=-1).bmm(self.embeddings[3](x).reshape(bs, -1, h*w).permute(0, 2, 1)) + nl
        
        nl = nl.mean(dim=1)
        # x = x + self.embeddings[4](nl).reshape(bs, -1, 1, 1)
        alpha, beta = self.se(nl).split(self.channel, dim=1)
        x = x * alpha.reshape(bs, self.channel, 1, 1) + beta.reshape(bs, self.channel, 1, 1)
        return x


class MyFilm(nn.Module):
    def __init__(self, cfg, num_words, padding_idx, norm_layer, num_colors, num_types):
        super().__init__()
        if norm_layer == None:
            norm_layer = nn.BatchNorm2d
        self.rnn = RNN(cfg, num_words, padding_idx)
        h, w = cfg.DATA.GLOBAL_SIZE
        cnn = models.resnet50(pretrained=True, norm_layer=norm_layer)
        cnn.layer4[0].conv2.stride = 1
        cnn.layer4[0].downsample[0].stride = 1

        self.stem = nn.Sequential(
            cnn.conv1, cnn.bn1, cnn.relu, cnn.maxpool
        )
        self.film1 = FilmBlock(cnn.layer1, cfg.MODEL.RNN.HIDDEN, 64, 64 * 4, h//2, w//2, spatial=True, has_pos=True)
        self.film2 = FilmBlock(cnn.layer2, cfg.MODEL.RNN.HIDDEN, 64 * 4, 128 * 4, h//4, w//4, spatial=True, has_pos=True)
        self.film3 = FilmBlock(cnn.layer3, cfg.MODEL.RNN.HIDDEN, 128 * 4, 256 * 4, h//8, w//8, spatial=True, has_pos=True)
        self.film4 = FilmBlock(cnn.layer4, cfg.MODEL.RNN.HIDDEN, 256 * 4, 512 * 4, h//8, w//8, spatial=True, has_pos=True)

        self.out = nn.Sequential(
            nn.Conv2d(2048, 1024, 3, padding=1), norm_layer(1024), nn.ReLU(True),
            nn.Conv2d(1024, 512, 3, padding=1), norm_layer(512), nn.ReLU(True),
            nn.Conv2d(512, 256, 3, padding=1), norm_layer(256), nn.ReLU(True),
            nn.Conv2d(256, 1, 3, padding=1))
        
        if norm_layer == nn.BatchNorm2d:
            norm_layer = nn.BatchNorm1d

        self.color = nn.Sequential(
            norm_layer(2048), nn.Linear(2048, num_colors),
            # norm_layer(1024), nn.ReLU(True),
            # nn.Linear(1024, 512), norm_layer(512), nn.ReLU(True),
            # nn.Linear(512, num_colors)
        )
        self.types = nn.Sequential(
            norm_layer(2048), nn.Linear(2048, num_types),
            # norm_layer(1024), nn.ReLU(True),
            # nn.Linear(1024, 512), norm_layer(512), nn.ReLU(True),
            # nn.Linear(512, num_types)
        )

    def forward(self, nl, img, activation_map=None):
        nl = self.rnn(nl)
        img = self.stem(img)
        img = self.film1(img, nl)
        img = self.film2(img, nl)
        img = self.film3(img, nl)
        img = self.film4(img, nl)

        if not self.training:
            return self.out(img)#.sigmoid()
        else:
            pred_map = self.out(img)
            vectors = (img * activation_map).sum(dim=(2, 3)) / activation_map.sum(dim=(2, 3))
            color = self.color(vectors)
            types = self.types(vectors)
            return pred_map, color, types