import glob
import os
import pickle
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data_root,
                 transform=None,
                 mode='train'
                 ):

        super().__init__()
        self.transform = transform
        self.mode = mode
        self.segments = glob.glob(os.path.join(data_root, "*.npz"))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        segment_path = self.segments[index]
        data = np.load(segment_path)
        wav_base = data["wav_base_features"]
        wav_emotion = data["wav_emotion_features"]
        arcface = data["arcface_features"]
        emotion = data["emotion_features"]
        affecnet8 = data["affecnet8_features"]
        rafdb = data["rafdb_features"]
        valences = data["valences"]
        arousals = data["arousals"]

        return wav_base, wav_emotion, arcface, emotion, affecnet8, rafdb, valences, arousals


if __name__ == "__main__":

    train_dataset = CustomDataset(
        "/data1/zww/abaw/VA_splits_w300_s200/Train_Set")
    features = ["wav_base", "wav_emotion",
                "arcface", "emotion", "affecnet8", "rafdb"]

    mean_std_dict = {feature: {"mean": None, "std": None}
                     for feature in features}

    for feature in tqdm(features):
        feature_list = []
        for wav_base, wav_emotion, arcface, emotion, affecnet8, rafdb, valences, arousals in tqdm(train_dataset):
            x = {"wav_base": wav_base, "wav_emotion": wav_emotion,
                    "arcface": arcface, "emotion": emotion, "affecnet8": affecnet8, "rafdb": rafdb}
            feature_list.append(x[feature])
        
        feature_list = np.concatenate(feature_list,axis=0)
        mean_std_dict[feature]["mean"] = np.mean(feature_list,axis=0)
        mean_std_dict[feature]["std"] = np.std(feature_list,axis=0)


    with open("features_mean_std.pkl","wb") as f:
        pickle.dump(mean_std_dict,f)