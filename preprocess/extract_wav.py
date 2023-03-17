import sys
import os
from tqdm import tqdm
wav_dir = "wav"

os.makedirs(wav_dir,exist_ok=True)

videos = []
for d in os.listdir("video"):
    for b in os.listdir(os.path.join("video",d)):
        videos.append(os.path.join("video",d,b))
for v in tqdm(videos):
    os.system("ffmpeg -i {} -f wav -ar 16000 {}".format(v,os.path.join(wav_dir,os.path.basename(v).split(".")[0]+".wav")))