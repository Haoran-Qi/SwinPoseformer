import pickle
from os import listdir
from os.path import isfile, join
import mmcv
import pdb

f = "../gym_validate/val_org_final.pkl"
file = open(f, 'rb')
data = pickle.load(file)
pdb.set_trace()