# %%
import os
import numpy as np
from tqdm import tqdm

# %%
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

# %%
dest_dir = "/data1/zww/abaw/VA_splits_w300_s200"
src_dir = "/data1/zww/abaw/VA_samples3"
# window = 300
# stride = 200
window = 300
stride = 200
mods  = os.listdir(src_dir)
for mod in mods:
    save_dir = os.path.join(dest_dir,mod)
    os.makedirs(save_dir,exist_ok=True)
    samples_dir = os.path.join(src_dir,mod)
    samples = os.listdir(samples_dir)
    for sample in tqdm(samples):
        data = np.load(os.path.join(samples_dir,sample))
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


