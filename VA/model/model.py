import torch
import torch.nn as nn
from .trans_encoder import TransEncoder
from .tcn import TemporalConvNet

# 'wav_base','wav_emotion','arcface','emotion','affecnet8','rafdb'
# 'erasing'没用, arcface0.3左右，emotion0.54, wav_base 0.25左右


class Model(nn.Module):
    def __init__(self, cfg, modality=['wav_base', 'wav_emotion', 'arcface', 'emotion', 'affecnet8', 'rafdb'],
                 embedding_dim={'wav_base': 512, 'wav_emotion': 512,
                                'arcface': 512, 'emotion': 1280, 'affecnet8': 512, 'rafdb': 512},
                 tcn_channel={
                     'wav_base': [512, 256, 128],
                     'wav_emotion': [512, 256, 128],
                     'arcface': [512, 256, 128],
                     'emotion': [1024, 512, 256],
                     'affecnet8': [512, 256, 128],
                     'rafdb': [512, 256, 128]
    }):
        super(Model, self).__init__()
        self.modality = modality

        self.temporal, self.fusion = nn.ModuleDict(), None

        for modal in self.modality:
            self.temporal[modal] = TemporalConvNet(num_inputs=embedding_dim[modal],
                                                   num_channels=tcn_channel[modal],
                                                   kernel_size=cfg.Model.kernel_size, dropout=cfg.Solver.dropout, attention=False)

        conv_dim = 0
        for m in self.modality:
            conv_dim += tcn_channel[m][-1]

        self.decoder = TransEncoder(
            inc=conv_dim, outc=512, dropout=cfg.Solver.dropout, nheads=cfg.Model.num_head, nlayer=cfg.Model.num_layer)
        self.vhead = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256, cfg.Model.bin_num),
        )
        self.ahead = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Linear(256,  cfg.Model.bin_num),
        )

    def forward(self, x):
        # wav_base, b*l*512; wav_emotion, b*l*512; arcface, b*l*512; emotion, b*l*1280; erasing, b*l*2048

        bs, seq_len, _ = x[self.modality[0]].shape
        for m in self.modality:
            x[m] = x[m].transpose(1, 2)
            x[m] = self.temporal[m](x[m])

        feat_list = []
        for m in self.modality:
            feat_list.append(x[m])
        feat = torch.cat(feat_list, dim=1)

        out = self.decoder(feat)

        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        vout = self.vhead(out)
        aout = self.ahead(out)
        return vout, aout
