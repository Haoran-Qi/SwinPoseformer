import torch
import pickle
import pdb

from mmaction.apis import init_recognizer, inference_recognizer

config_file = '/home/haoran/Video-Swin-Transformer/configs/skeleton/stgcn/stgcn_803_gym_rsn_keypoint.py'
# download the checkpoint from model zoo and put it in `checkpoints/`
checkpoint_file = '/home/haoran/Video-Swin-Transformer/work_dirs/stgcn/stgcn_80e_gym_rsn_keypoint/epoch_80.pth'

# assign the desired device.
device = 'cuda:0' # or 'cpu'
device = torch.device(device)

 # build the model from a config file and a checkpoint file
model = init_recognizer(config_file, checkpoint_file, device=device)

# load pickle file
file = open("/home/haoran/Video-Swin-Transformer/data/gym/rsn_pose/val_org_final.pkl",'rb')
annoVal = pickle.load(file)
file.close()

pdb.set_trace()


# test a single video and show the result:
video = 'demo/demo.mp4'
labels = 'tools/data/kinetics/label_map_k400.txt'
results = inference_recognizer(model, video)

# show the results
labels = open('tools/data/kinetics/label_map_k400.txt').readlines()
labels = [x.strip() for x in labels]
results = [(labels[k[0]], k[1]) for k in results]

print(f'The top-5 labels with corresponding scores are:')
for result in results:
    print(f'{result[0]}: ', result[1])