# %%
import os
import numpy as np
from tqdm import tqdm
import cv2
from scipy.interpolate import interp2d
import torch.nn.functional as F
import torch
import pandas as pd

label_root_dir = "label/AU_Detection_Challenge/"
video_dir = "video"
features_dir = "features"
align_dir = "align"
sample_root_dir = "/data1/zww/abaw/AU_samples1"

wav_base_dir = os.path.join(features_dir, "wav2vec2-base-960h")
wav_emotion_dir = os.path.join(
    features_dir, "wav2vec2-large-robust-12-ft-emotion-msp-dim")
arcface_dir = os.path.join(features_dir, "arcface")
rafdb_dir = os.path.join(features_dir, "rafdb")
affecnet8_dir = os.path.join(features_dir, "affecnet8")
emotion_dir = os.path.join(features_dir, "face_emotion_recognition")


mods = os.listdir(label_root_dir)

for mod in mods:
    label_dir = os.path.join(label_root_dir, mod)
    label_files = os.listdir(label_dir)

    sample_dir = os.path.join(sample_root_dir, mod)
    os.makedirs(sample_dir, exist_ok=True)

    for label_file in tqdm(label_files):
        video_path = os.path.join(video_dir, label_file.replace(
            "_left", "").replace("_right", "").replace(".txt", ".mp4"))
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, label_file.replace(
                "_left", "").replace("_right", "").replace(".txt", ".avi"))
        video = cv2.VideoCapture(video_path)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps = video.get(cv2.CAP_PROP_FPS)
        # duration = int(num_frames / fps)
        # num_frames = int(fps*duration)
        print("num_frames", num_frames)

        # wav_base
        wav_base_feature_file = os.path.join(
            wav_base_dir, label_file.replace(
                "_left", "").replace("_right", "").replace(".txt", ".npz"))
        wav_base_feature = np.load(wav_base_feature_file)["arr_0"]
        wav_base_feature = torch.tensor(wav_base_feature).unsqueeze(0)
        interp_wav_base_feature = F.interpolate(
            wav_base_feature, [num_frames, wav_base_feature.shape[-1]], mode="nearest")
        interp_wav_base_feature = interp_wav_base_feature.squeeze(
            0).squeeze(0).numpy()

        # wav emotion
        wav_emotion_feature_file = os.path.join(
            wav_emotion_dir, label_file.replace(
                "_left", "").replace("_right", "").replace(".txt", ".npz"))
        wav_emotion_feature = np.load(wav_emotion_feature_file)["arr_0"]
        wav_emotion_feature = torch.tensor(wav_emotion_feature).unsqueeze(0)
        interp_wav_emotion_feature = F.interpolate(
            wav_emotion_feature, [num_frames, wav_emotion_feature.shape[-1]], mode="nearest")
        interp_wav_emotion_feature = interp_wav_emotion_feature.squeeze(
            0).squeeze(0).numpy()
        # 至此，音频特征长度和图片帧数一致

        # 读取图片范围
        img_dir = os.path.join(align_dir, label_file.split(".")[0])
        img_names = os.listdir(img_dir)
        img_names.sort()
        begin = int(img_names[0].split(".")[0]) - 1
        end = int(img_names[-1].split(".")[0]) - 1
        end = min(end, num_frames)

        # 读取标签
        labels = []
        with open(os.path.join(label_dir, label_file)) as f:
            lines = f.readlines()
            for line in lines[1:]:
                label = list(map(int,line.strip().split(",")))
                labels.append(label)
        labels = np.array(labels)

        # 读取图片特征
        # arcface
        img_arcface_feature_dir = os.path.join(
            arcface_dir, label_file.split(".")[0])
        prev_arcface_feature = None
        img_arcface_features = []
        for i in tqdm(range(begin, end)):
            # 注意我们的帧数从0开始，保存的帧数从1开始
            img_arcface_feature_file = os.path.join(
                img_arcface_feature_dir, str(i+1).zfill(5)+".npy")
            if os.path.exists(img_arcface_feature_file):
                img_arcface_feature = np.load(img_arcface_feature_file)[0]
                prev_arcface_feature = img_arcface_feature
            else:
                img_arcface_feature = prev_arcface_feature
            img_arcface_features.append(img_arcface_feature)
        img_arcface_features = np.stack(
            img_arcface_features)  # 从begin到end的所有帧的特征
        print("img_arcface_features", img_arcface_features.shape)
        # emotion
        img_emotion_feature_dir = os.path.join(
            emotion_dir, label_file.split(".")[0])
        prev_emotion_feature = None
        img_emotion_features = []
        for i in tqdm(range(begin, end)):
            # 注意我们的帧数从0开始，保存的帧数从1开始
            img_emotion_feature_file = os.path.join(
                img_emotion_feature_dir, str(i+1).zfill(5)+".npy")
            if os.path.exists(img_emotion_feature_file):
                img_emotion_feature = np.load(img_emotion_feature_file)
                prev_emotion_feature = img_emotion_feature
            else:
                img_emotion_feature = prev_emotion_feature
            img_emotion_features.append(img_emotion_feature)
        img_emotion_features = np.stack(
            img_emotion_features)  # 从begin到end的所有帧的特征
        print("img_emotion_features", img_emotion_features.shape)

        # rafdb
        img_rafdb_feature_dir = os.path.join(
            rafdb_dir, label_file.split(".")[0])
        prev_rafdb_feature = None
        img_rafdb_features = []

        for i in tqdm(range(begin, end)):
            # 注意我们的帧数从0开始，保存的帧数从1开始
            img_rafdb_feature_file = os.path.join(
                img_rafdb_feature_dir, str(i+1).zfill(5)+".npy")
            if os.path.exists(img_rafdb_feature_file):
                img_rafdb_feature = np.load(img_rafdb_feature_file)
                prev_rafdb_feature = img_rafdb_feature
            else:
                img_rafdb_feature = prev_rafdb_feature
            img_rafdb_features.append(img_rafdb_feature)
        img_rafdb_features = np.stack(
            img_rafdb_features)  # 从begin到end的所有帧的特征
        print("img_rafdb_features", img_rafdb_features.shape)


        # affecnet8
        img_affecnet8_feature_dir = os.path.join(
            affecnet8_dir, label_file.split(".")[0])
        prev_affecnet8_feature = None
        img_affecnet8_features = []

        for i in tqdm(range(begin, end)):
            # 注意我们的帧数从0开始，保存的帧数从1开始
            img_affecnet8_feature_file = os.path.join(
                img_affecnet8_feature_dir, str(i+1).zfill(5)+".npy")
            if os.path.exists(img_affecnet8_feature_file):
                img_affecnet8_feature = np.load(img_affecnet8_feature_file)
                prev_affecnet8_feature = img_affecnet8_feature
            else:
                img_affecnet8_feature = prev_affecnet8_feature
            img_affecnet8_features.append(img_affecnet8_feature)
        img_affecnet8_features = np.stack(
            img_affecnet8_features)  # 从begin到end的所有帧的特征
        print("img_affecnet8_features", img_affecnet8_features.shape)

        print("begin,end",begin,end)

        select_wav_base_features = interp_wav_base_feature[begin:end]
        print("select_wav_base_features", select_wav_base_features.shape)

        select_wav_emotion_features = interp_wav_emotion_feature[begin:end]
        print("select_wav_emotion_features", select_wav_emotion_features.shape)

        select_arcface_features = img_arcface_features
        print("select_arcface_features", select_arcface_features.shape)

        select_emotion_features = img_emotion_features
        print("select_emotion_features", select_emotion_features.shape)

        select_rafdb_features = img_rafdb_features
        print("select_rafdb_features", select_rafdb_features.shape)

        select_affecnet8_features = img_affecnet8_features
        print("select_affecnet8_features", select_affecnet8_features.shape)

        select_labels = np.array(labels[begin:end])

        print("select_labels", len(select_labels))

        assert len(select_wav_base_features) == len(select_arcface_features),[label_file,len(select_wav_base_features),len(select_arcface_features)]

        save_path = os.path.join(sample_dir, label_file.split(".")[0]+".npz")

        np.savez(save_path, select_wav_base_features=select_wav_base_features, select_wav_emotion_features=select_wav_emotion_features,
                 select_arcface_features=select_arcface_features, select_emotion_features=select_emotion_features, select_rafdb_features=select_rafdb_features,
                 select_affecnet8_features=select_affecnet8_features,select_labels=select_labels)

        print("-"*20)
