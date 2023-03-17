import os
import random
import numpy as np
import torch
import yaml
from munch import DefaultMunch
from tqdm import tqdm
import pickle
from model import Model
from utils.loss import *
import sys

if __name__ == '__main__':
    device = torch.device('cuda')

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    config_path = 'config/config.yml'
    yaml_dict = yaml.load(
        open(config_path, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    cfg = DefaultMunch.fromDict(yaml_dict)

    pretrained_path = sys.argv[1]

    save_path = sys.argv[2]

    model = Model(cfg, cfg.Model.modality)

    print("loading from:", pretrained_path)
    pretrain_dict = torch.load(pretrained_path)
    model_dict = model.state_dict()
    pretrain_dict = {k.replace('module.', '')                         : v for k, v in pretrain_dict.items()}
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    model = model.cuda()
    model.eval()

    window = 300
    stride = 300
    video_frames_dict = {}

    # 获取每个视频帧数
    with open("/data2/zww/abaw/testset/test_set_examples/CVPR_5th_ABAW_EXPR_test_set_example.txt","r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            t = line.split("/")[0]
            if t not in video_frames_dict:
                video_frames_dict[t] = 1
            else:
                video_frames_dict[t] += 1
    print("videos:",len(video_frames_dict))


    # 获取视频列表
    test_label = "/data2/zww/abaw/testset/names_of_videos_in_each_test_set/Expression_Classification_Challenge_test_set_release.txt"
    test_videos = []
    with open(test_label,"r") as f:
        for line in f:
            test_videos.append(line.strip())

    # 读取mean_std
    with open("/data2/zww/abaw/preprocess/features_mean_std.pkl", "rb") as f:
        mean_std_dict = pickle.load(f)


    src_dir = "/data1/zww/abaw/EXPR_samples_testset"

    with open(save_path,"w") as f:
        f.write("image_location,Neutral,Anger,Disgust,Fear,Happiness,Sadness,Surprise,Other\n")
        for test_video in tqdm(test_videos):
            preds = []
            data = np.load(os.path.join(src_dir, test_video+".npz"))
            select_wav_base_features = data["select_wav_base_features"].astype(
                np.float32)
            select_wav_emotion_features = data["select_wav_emotion_features"].astype(
                np.float32)
            select_arcface_features = data["select_arcface_features"].astype(
                np.float32)
            select_emotion_features = data["select_emotion_features"].astype(
                np.float32)
            select_affecnet8_features = data["select_affecnet8_features"].astype(
                np.float32)
            select_rafdb_features = data["select_rafdb_features"].astype(
                np.float32)

            length = min(len(select_arcface_features),
                        len(select_wav_base_features))


            for i in range(length//stride+1):
                begin = stride * i
                if begin >= length:
                    break
                end = min(begin + window, length)

                wav_base_features = select_wav_base_features[begin:end]
                wav_base = (wav_base_features - mean_std_dict["wav_base"]
                            ["mean"])/mean_std_dict["wav_base"]["std"]

                wav_emotion_features = select_wav_emotion_features[begin:end]
                wav_emotion = (wav_emotion_features - mean_std_dict["wav_emotion"]
                            ["mean"])/mean_std_dict["wav_emotion"]["std"]

                arcface_features = select_arcface_features[begin:end]
                arcface = (arcface_features - mean_std_dict["arcface"]
                        ["mean"])/mean_std_dict["arcface"]["std"]

                emotion_features = select_emotion_features[begin:end]
                emotion = (emotion_features - mean_std_dict["emotion"]
                        ["mean"])/mean_std_dict["emotion"]["std"]

                affecnet8_features = select_affecnet8_features[begin:end]
                affecnet8 = (affecnet8_features - mean_std_dict["affecnet8"]
                            ["mean"])/mean_std_dict["affecnet8"]["std"]

                rafdb_features = select_rafdb_features[begin:end]
                rafdb = (rafdb_features - mean_std_dict["rafdb"]
                        ["mean"])/mean_std_dict["rafdb"]["std"]

                wav_base = torch.tensor(wav_base).to(device).unsqueeze(0)
                wav_emotion = torch.tensor(wav_emotion).to(device).unsqueeze(0)
                arcface = torch.tensor(arcface).to(device).unsqueeze(0)
                emotion = torch.tensor(emotion).to(device).unsqueeze(0)
                affecnet8 = torch.tensor(affecnet8).to(device).unsqueeze(0)
                rafdb = torch.tensor(rafdb).to(device).unsqueeze(0)

                x = {"wav_base": wav_base, "wav_emotion": wav_emotion,
                    "arcface": arcface, "emotion": emotion, "affecnet8": affecnet8, "rafdb": rafdb}
                with torch.no_grad():
                    outputs = model(x)

                preds += torch.argmax(outputs, 1).detach().cpu().numpy().tolist()


            frames_length = video_frames_dict[test_video]

            common_length = min(length,frames_length)

            # 填充共同部分
            for i in range(frames_length):
                f.write(test_video+"/"+str(i+1).zfill(5)+".jpg,"+str(preds[i])+"\n")





