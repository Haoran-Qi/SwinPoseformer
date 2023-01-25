import urllib.request
from clint.textui import progress
from tqdm import tqdm
import requests
import logging
import pdb

logging.basicConfig(filename='download.log', encoding='utf-8', level=logging.DEBUG)

nut60Files = [
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/125",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/126",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/127",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/128",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/129",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/130",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/131",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/132",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/133",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/134",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/135",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/136",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/137",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/138",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/139",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/140",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/141"
]

ntu120Files = [
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/142",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/143",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/144",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/145",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/146",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/147",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/148",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/149",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/150",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/151",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/152",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/153",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/154",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/155",
    "https://rose1.ntu.edu.sg/dataset/actionRecognition/download/156",

]

def download(url, path):
    # urllib.request.urlretrieve(url, path)
    res = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        # pdb.set_trace()
        total_length = int(r.headers.get('content-length'))
        for chunk in progress.bar(r.iter_content(chunk_size=1024), expected_size=(total_length/1024) + 1): 
            if chunk:
                f.write(chunk)
                f.flush()


print('Start Downloading NTU 60')
for i in tqdm(range(len(nut60Files))):
    fileName = "nturgbd_rgb_s" + str(i+1).zfill(3) + ".zip"
    path = "/home/haoran/ntu/ntu60/" + fileName
    logging.info('downloading ntu60 ' + fileName + " into file: "  + path)
    download(nut60Files[i], path)
    logging.info(path + "downloaded")


print('Start Downloading NTU 120')
for i in tqdm(range(len(nut60Files))):
    fileName = "nturgbd_rgb_s" + str(i+18).zfill(3) + ".zip"
    path = "/home/haoran/ntu/ntu120/" + fileName
    logging.info('downloading ntu60 ' + fileName + " into file: "  + path)
    download(nut60Files[i], path)
    logging.info(path + "downloaded")