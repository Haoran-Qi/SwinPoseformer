
import numpy as np
import urllib
from mmcv import load, dump
import pdb

gym_train_ann_file = '/home/haoran/Video-Swin-Transformer/data/posec3d/gym_train.pkl'
gym_val_ann_file = '/home/haoran/Video-Swin-Transformer/data/posec3d/gym_val.pkl'
lines = list(urllib.request.urlopen('https://sdolivia.github.io/FineGym/resources/dataset/gym99_categories.txt'))
gym_categories = [x.decode().strip().split('; ')[-1] for x in lines]
gym_annos = load(gym_train_ann_file) + load(gym_val_ann_file)
vid_path = "/home/haoran/Video-Swin-Transformer/data/gym/subactions/e3EsDlpNo0c_E_001952_001988_A_0018_0020.mp4"

annoo = [x for x in gym_annos if x['frame_dir'] == "e3EsDlpNo0c_001952_001988_0018_0020"][0]
annoo["bbox"] = np.zeros((annoo["total_frames"], 5))


cur = load("./example.pkl")
pdb.set_trace()