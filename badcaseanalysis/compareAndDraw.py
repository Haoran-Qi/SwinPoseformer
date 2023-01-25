import json
import numpy as np
import pickle
from matplotlib import pyplot
import pdb


# f = open('./data/xsub_test.json')
f = open('./data/rsn_test.json')
data = json.load(f)

top1 = [np.argmax(d) for d in data]


# load pickle file
# file = open("/home/haoran/Video-Swin-Transformer/data/posec3d/gym_val.pkl",'rb')
file = open("/home/haoran/Video-Swin-Transformer/data/gym/rsn_pose/val_org_final.pkl",'rb')
annoVal = pickle.load(file)
file.close()

combined = []

### save to json file

for i in range(len(annoVal)):
    combined.append(
        {
            'frame_dir': annoVal[i]['frame_dir'],
            'label': int(annoVal[i]['label']),
            'guess': int(top1[i])
        }
    )


with open("./data/rsn_test_compare.json", "w") as outfile:
    json.dump(combined, outfile)


### plot a histogram
top1 = [int(x) for x in top1]
labels = [int(x['label']) for x in annoVal]
bins = np.arange(0,100)

pyplot.hist(top1, bins, alpha=0.5, label='guess')
pyplot.hist(labels, bins, alpha=0.5, label='labels')
pyplot.legend(loc='upper left')
pyplot.show()


