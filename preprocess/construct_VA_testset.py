import os
import numpy as np
from tqdm import tqdm
import cv2
from scipy.interpolate import interp2d
import torch.nn.functional as F
import torch
import pandas as pd
import pickle


video_frames_dict = {}
with open("/data2/zww/abaw/testset/test_set_examples/CVPR_5th_ABAW_VA_test_set_example.txt","r") as f:
    lines = f.readlines()
    for line in lines[1:]:
        t = line.split("/")[0]
        if t not in video_frames_dict:
            video_frames_dict[t] = 1
        else:
            video_frames_dict[t] += 1


with open("video_frames_dict_VA.pkl","wb") as f:
    pickle.dump(video_frames_dict,f)


test_label = "/data2/zww/abaw/testset/names_of_videos_in_each_test_set/Valence_Arousal_Estimation_Challenge_test_set_release.txt"

video_dir = "video"
features_dir = "features"
align_dir = "align"


sample_root_dir = "/data1/zww/abaw/VA_samples_testset"

wav_base_dir = os.path.join(features_dir, "wav2vec2-base-960h")
wav_emotion_dir = os.path.join(
    features_dir, "wav2vec2-large-robust-12-ft-emotion-msp-dim")
arcface_dir = os.path.join(features_dir, "arcface")
rafdb_dir = os.path.join(features_dir, "rafdb")
affecnet8_dir = os.path.join(features_dir, "affecnet8")
emotion_dir = os.path.join(features_dir, "face_emotion_recognition")

os.makedirs(sample_root_dir, exist_ok=True)

video_files = []
with open(test_label,"r") as f:
    for line in f:
        video_files.append(line.strip())


for video_file in tqdm(video_files):

    len_label_frames = video_frames_dict[video_file]

    # wav_base
    wav_base_feature_file = os.path.join(
        wav_base_dir, video_file.replace(
            "_left", "").replace("_right", "")+".npz")

    wav_base_feature = np.load(wav_base_feature_file)["arr_0"]
    wav_base_feature = torch.tensor(wav_base_feature).unsqueeze(0)
    interp_wav_base_feature = F.interpolate(
        wav_base_feature, [len_label_frames, wav_base_feature.shape[-1]], mode="nearest")
    interp_wav_base_feature = interp_wav_base_feature.squeeze(
        0).squeeze(0).numpy()

    # wav emotion
    wav_emotion_feature_file = os.path.join(
        wav_emotion_dir, video_file.replace(
            "_left", "").replace("_right", "")+".npz")
    wav_emotion_feature = np.load(wav_emotion_feature_file)["arr_0"]
    wav_emotion_feature = torch.tensor(wav_emotion_feature).unsqueeze(0)
    interp_wav_emotion_feature = F.interpolate(
        wav_emotion_feature, [len_label_frames, wav_emotion_feature.shape[-1]], mode="nearest")
    interp_wav_emotion_feature = interp_wav_emotion_feature.squeeze(
        0).squeeze(0).numpy()
    # 至此，音频特征长度和图片帧数一致


    # 读取图片特征
    # arcface
    img_arcface_feature_dir = os.path.join(
        arcface_dir, video_file.split(".")[0])
    img_arcface_features = np.zeros((len_label_frames,512),dtype=np.float32)
    for i in tqdm(range(len_label_frames)):
        # 注意我们的帧数从0开始，保存的帧数从1开始
        img_arcface_feature_file = os.path.join(
            img_arcface_feature_dir, str(i+1).zfill(5)+".npy")
        if os.path.exists(img_arcface_feature_file):
            img_arcface_feature = np.load(img_arcface_feature_file)[0]
            img_arcface_features[i] = img_arcface_feature
    
    print("img_arcface_features", img_arcface_features.shape)
    # emotion
    img_emotion_feature_dir = os.path.join(
        emotion_dir, video_file.split(".")[0])
    img_emotion_features = np.zeros((len_label_frames,1280),dtype=np.float32)
    for i in tqdm(range(len_label_frames)):
        # 注意我们的帧数从0开始，保存的帧数从1开始
        img_emotion_feature_file = os.path.join(
            img_emotion_feature_dir, str(i+1).zfill(5)+".npy")
        if os.path.exists(img_emotion_feature_file):
            img_emotion_feature = np.load(img_emotion_feature_file)
            img_emotion_features[i] = img_emotion_feature
    print("img_emotion_features", img_emotion_features.shape)

    # rafdb
    img_rafdb_feature_dir = os.path.join(
        rafdb_dir, video_file.split(".")[0])
    img_rafdb_features = np.zeros((len_label_frames,512),dtype=np.float32)

    for i in tqdm(range(len_label_frames)):
        # 注意我们的帧数从0开始，保存的帧数从1开始
        img_rafdb_feature_file = os.path.join(
            img_rafdb_feature_dir, str(i+1).zfill(5)+".npy")
        if os.path.exists(img_rafdb_feature_file):
            img_rafdb_feature = np.load(img_rafdb_feature_file)
            img_rafdb_features[i] = img_rafdb_feature
    print("img_rafdb_features", img_rafdb_features.shape)
    # affecnet8
    img_affecnet8_feature_dir = os.path.join(
        affecnet8_dir, video_file.split(".")[0])
    img_affecnet8_features = np.zeros((len_label_frames,512),dtype=np.float32)
    for i in tqdm(range(len_label_frames)):
        # 注意我们的帧数从0开始，保存的帧数从1开始
        img_affecnet8_feature_file = os.path.join(
            img_affecnet8_feature_dir, str(i+1).zfill(5)+".npy")
        if os.path.exists(img_affecnet8_feature_file):
            img_affecnet8_feature = np.load(img_affecnet8_feature_file)
            img_affecnet8_features[i] = img_affecnet8_feature



    select_wav_base_features = interp_wav_base_feature
    print("select_wav_base_features", select_wav_base_features.shape)

    select_wav_emotion_features = interp_wav_emotion_feature
    print("select_wav_emotion_features", select_wav_emotion_features.shape)

    select_arcface_features = img_arcface_features
    print("select_arcface_features", select_arcface_features.shape)

    select_emotion_features = img_emotion_features
    print("select_emotion_features", select_emotion_features.shape)

    select_rafdb_features = img_rafdb_features
    print("select_rafdb_features", select_rafdb_features.shape)

    select_affecnet8_features = img_affecnet8_features
    print("select_affecnet8_features", select_affecnet8_features.shape)
    
    save_path = os.path.join(sample_root_dir, video_file.split(".")[0]+".npz")

    np.savez(save_path, select_wav_base_features=select_wav_base_features, select_wav_emotion_features=select_wav_emotion_features,
                select_arcface_features=select_arcface_features, select_emotion_features=select_emotion_features, select_rafdb_features=select_rafdb_features,
                select_affecnet8_features=select_affecnet8_features)

    print("-"*20)
