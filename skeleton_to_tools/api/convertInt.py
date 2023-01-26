import pickle
from os import listdir
from os.path import isfile, join
import mmcv
import pdb


f = "../gym_validate/val_org_bad_final.pkl"
file = open(f, 'rb')
data = pickle.load(file)
# pdb.set_trace()



# remove empty data

newData = []
count = 0
for instance in data:
    if "label"  in instance.keys():
        newData.append(instance)
        count += 1
# pdb.set_trace()

for instance in newData:
    # if "label" not in instance.keys():
        # pdb.set_trace()
    instance["label"] = int(instance["label"])

mmcv.dump(newData, "../gym_validate/val_org_final.pkl")