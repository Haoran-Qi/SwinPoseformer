import pickle
from os import listdir
from os.path import isfile, join
import mmcv
import pdb


onlyfiles = [join("./gym_train", f) for f in listdir("./gym_train") if isfile(join("./gym_train", f))]

one = []
for f in onlyfiles:
    
    one = one + datafile = open(f, 'rb')
    data = pickle.load(file)

# fixing wrong label type: str --> int
for instance in one:
    instance["label"] = int(instance["label"])

pdb.set_trace()

mmcv.dump(one, "./gym_train/gym_train.pkl")
