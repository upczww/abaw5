import torch
import torch.nn as nn
import yaml
from munch import DefaultMunch

from .tcn import TemporalConvNet
from .trans_encoder import TransEncoder


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
        self.encoder = TransEncoder(
            inc=conv_dim, outc=cfg.Model.out_dim, dropout=cfg.Solver.dropout, nheads=cfg.Model.num_head, nlayer=cfg.Model.num_layer)
        self.head = nn.Sequential(
            nn.Linear(cfg.Model.out_dim, cfg.Model.out_dim//2),
            nn.BatchNorm1d(cfg.Model.out_dim//2),
            nn.Linear(cfg.Model.out_dim//2, 12),
        )

    def forward(self, x):
        # wav_base, b*l*512; wav_emotion, b*l*512; arcface, b*l*512; emotion, b*l*1280; 

        bs, seq_len, _ = x[self.modality[0]].shape
        for m in self.modality:
            x[m] = x[m].transpose(1, 2)
            x[m] = self.temporal[m](x[m])

        feat_list = []
        for m in self.modality:
            feat_list.append(x[m])
        out = torch.cat(feat_list, dim=1)
        out = self.encoder(out)

        out = torch.transpose(out, 1, 0)
        out = torch.reshape(out, (bs*seq_len, -1))

        out = self.head(out)
        return out


if __name__ == "__main__":
    config_path = 'config/config.yml'
    yaml_dict = yaml.load(
        open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    cfg = DefaultMunch.fromDict(yaml_dict)
    model = Model(cfg)

    print(model)
