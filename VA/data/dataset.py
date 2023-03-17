import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import json
from PIL import Image
import torch
from torchvision import transforms
import torchaudio
import torchaudio.transforms as audio_transforms
import random
import glob
import pickle


def fill_nearest(matrix):
    # 找出第一列为0的行
    row_index = np.where(matrix[:, 0] == 0)[0]
    # 找出第一列非0的行
    row_index_not_zero = np.where(matrix[:, 0] != 0)[0]
    for row in row_index:
        # 将第一列为0的行用最近邻的第一列非0的行填充
        # 找出最近邻的非0行的索引
        nearest_row_index = min(row_index_not_zero, key=lambda x: abs(x - row))
        # 将最近邻的非0行的值赋给全0的行
        matrix[row] = matrix[nearest_row_index]
    return matrix


class CustomDataset(Dataset):
    def __init__(self, data_root,
                 transform=None,
                 mode='train'
                 ):

        super().__init__()
        self.transform = transform
        self.mode = mode
        self.segments = glob.glob(os.path.join(data_root, "*.npz"))
        self.segments.sort()

        with open("/data2/zww/abaw/preprocess/features_mean_std.pkl", "rb") as f:
            self.mean_std_dict = pickle.load(f)

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment_path = self.segments[index]
        # print(segment_path)
        data = np.load(segment_path)

        wav_base = data["wav_base_features"]
        wav_emotion = data["wav_emotion_features"]
        arcface = data["arcface_features"]
        emotion = data["emotion_features"]
        affecnet8 = data["affecnet8_features"]
        rafdb = data["rafdb_features"]


        valences = data["valences"]
        arousals = data["arousals"]

        wav_base = (wav_base - self.mean_std_dict["wav_base"]
                    ["mean"])/self.mean_std_dict["wav_base"]["std"]
        wav_emotion = (wav_emotion - self.mean_std_dict["wav_emotion"]
                       ["mean"])/self.mean_std_dict["wav_emotion"]["std"]

        emotion = (emotion - self.mean_std_dict["emotion"]
                   ["mean"])/self.mean_std_dict["emotion"]["std"]
        affecnet8 = (affecnet8 - self.mean_std_dict["affecnet8"]
                     ["mean"])/self.mean_std_dict["affecnet8"]["std"]
        rafdb = (rafdb - self.mean_std_dict["rafdb"]
                 ["mean"])/self.mean_std_dict["rafdb"]["std"]
        arcface = (arcface - self.mean_std_dict["arcface"]
                   ["mean"])/self.mean_std_dict["arcface"]["std"]
        if self.mode != "test":
            valences = data["valences"]
            arousals = data["arousals"]

            return wav_base.astype(np.float32), wav_emotion.astype(np.float32), arcface.astype(np.float32), emotion.astype(np.float32), affecnet8.astype(np.float32), rafdb.astype(np.float32), valences, arousals
        else:
            return wav_base.astype(np.float32), wav_emotion.astype(np.float32), arcface.astype(np.float32), emotion.astype(np.float32), affecnet8.astype(np.float32), rafdb.astype(np.float32)


def get_loader(cfg):

    train_dataset = CustomDataset(cfg.Data.train_data_root)
    valid_dataset = CustomDataset(cfg.Data.val_data_root)

    train_loader = DataLoader(train_dataset, batch_size=cfg.Data.loader.batch_size,
                              num_workers=cfg.Data.loader.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    if cfg.Data.loader.test_batch_size:
        test_batch_size = cfg.Data.loader.test_batch_size
    else:
        test_batch_size = cfg.Data.loader.batch_size
    valid_loader = DataLoader(valid_dataset, batch_size=test_batch_size,
                              num_workers=cfg.Data.loader.num_workers, shuffle=False, pin_memory=True,  drop_last=True)  # fixme [True|False]
    return train_loader, valid_loader
