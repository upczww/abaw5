
import os
import numpy as np
import glob

from tqdm import tqdm
from sklearn.model_selection import KFold


def split_sample(length,window=30,stride=15):
    splits = []
    for i in range(length//stride):
        begin = stride * i
        end = begin + window
        if end > length:
            begin = length - window
            end = length
            break
        splits.append([begin,end])
    return splits

dest_dir = "/data1/zww/abaw/VA_splits_w300_s200"
src_dir = "/data1/zww/abaw/VA_samples3"
window = 300
stride = 200
all_samples = np.array(glob.glob(os.path.join(src_dir,"**/*.npz")))


kf = KFold(n_splits=4)

for idx,(train_samples,val_samples) in enumerate(kf.split(all_samples)):
    print("#",idx)
    fold_dir = os.path.join(dest_dir+"_fold_"+str(idx+1))

    modes_samples = {"Train_Set":train_samples,"Validation_Set":val_samples}

    for mode in modes_samples:
        save_dir = os.path.join(fold_dir,mode)
        print("save_dir:",save_dir)
        os.makedirs(save_dir,exist_ok=True)
        samples_idx = modes_samples[mode]
        samples = all_samples[samples_idx]
        for sample in tqdm(samples):
            data = np.load(sample)
            select_wav_base_features=data["select_wav_base_features"]
            select_wav_emotion_features=data["select_wav_emotion_features"]
            select_arcface_features=data["select_arcface_features"]
            select_emotion_features=data["select_emotion_features"]
            select_affecnet8_features=data["select_affecnet8_features"]
            select_rafdb_features=data["select_rafdb_features"]

            select_valences=data["select_valences"]
            select_arousals=data["select_arousals"]
            if len(select_wav_base_features) != len(select_arcface_features):
                print([sample,len(select_wav_base_features),len(select_arcface_features)])
            length = len(select_wav_base_features)
            splits = split_sample(length,window,stride)
            for idx,s in enumerate(splits):
                b = s[0]
                e = s[1]
                wav_base_features = select_wav_base_features[b:e]
                wav_emotion_features = select_wav_emotion_features[b:e]
                arcface_features = select_arcface_features[b:e]
                emotion_features = select_emotion_features[b:e]
                affecnet8_features = select_affecnet8_features[b:e]
                rafdb_features = select_rafdb_features[b:e]

                valences = select_valences[b:e]
                arousals = select_arousals[b:e]

                if len(valences) != window:
                    continue
                save_path = os.path.join(save_dir,os.path.basename(sample).split(".")[0]+"_"+str(idx).zfill(4)+".npz")
                np.savez(save_path,wav_base_features=wav_base_features,wav_emotion_features=wav_emotion_features,\
                    arcface_features=arcface_features,emotion_features=emotion_features,\
                    affecnet8_features=affecnet8_features,rafdb_features=rafdb_features,\
                    valences=valences,arousals=arousals)


