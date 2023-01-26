# Copyright (c) OpenMMLab. All rights reserved.
import abc
import argparse
import os
import os.path as osp
from os import listdir
from os.path import isfile, join
import random as rd
import shutil
import string
from collections import defaultdict

import cv2
import mmcv
import numpy as np
import math

import pdb

try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')

try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

mmdet_root = ''
mmpose_root = ''

args = abc.abstractproperty()
args.det_config = f'~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = './faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = f'~/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xrsn50_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = './3xrsn50_coco_256x192-58f57a68_20201127.pth'  # noqa: E501


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def extract_frame(video_path):
    dname = gen_id()
    os.makedirs(dname, exist_ok=True)
    frame_tmpl = osp.join(dname, 'img_{:05d}.jpg')
    vid = cv2.VideoCapture(video_path)
    size = (vid.get(3), vid.get(4))
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    while flag:
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return size, frame_paths


def detection_inference(args, frame_paths):
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def intersection(b0, b1):
    l, r = max(b0[0], b1[0]), min(b0[2], b1[2])
    u, d = max(b0[1], b1[1]), min(b0[3], b1[3])
    return max(0, r - l) * max(0, d - u)


def iou(b0, b1):
    i = intersection(b0, b1)
    u = area(b0) + area(b1) - i
    return i / u


def area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def removedup(bbox):
    def inside(box0, box1, thre=0.8):
        return intersection(box0, box1) / area(box0) > thre

    num_bboxes = bbox.shape[0]
    if num_bboxes == 1 or num_bboxes == 0:
        return bbox
    valid = []
    for i in range(num_bboxes):
        flag = True
        for j in range(num_bboxes):
            if i != j and inside(bbox[i],
                                 bbox[j]) and bbox[i][4] <= bbox[j][4]:
                flag = False
                break
        if flag:
            valid.append(i)
    return bbox[valid]

def gym_keep_one(bbox):
    
    audience_threshold = 0.3

    # find the largest bbox in each frame
    candidates = []
    for b in bbox:
        if len(b) == 0:
            candidates.append(np.zeros(5))
        else:
            areas = [area(x) for x in b]
            candidates.append(b[areas.index(max(areas))])

    # drop out missing frames based on the size of bbox
    avg_area = np.mean(np.array([area(x) for x in candidates]))
    candidates = [x if area(x) > audience_threshold* avg_area else None for x in candidates]

    # using adjancent value fill up missing frames
    first_not_none = next(x for x in candidates if x is not None)
    candidates[0] = first_not_none if candidates[0] is None else candidates[0]
    for i in range(len(candidates)):
        candidates[i] = candidates[i-1] if candidates[i] is None else candidates[i]
    return np.array(candidates)

def gym_det_postproc(det_results):
    det_results = [removedup(x) for x in det_results] 
    
    keepOne = gym_keep_one(det_results)
    return np.reshape(keepOne, (keepOne.shape[0], 1, keepOne.shape[1] ))

def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    num_frame = len(det_results)
    num_person = max([len(x) for x in det_results])
    kp = np.zeros((num_person, num_frame, 17, 3), dtype=np.float32)

    for i, (f, d) in enumerate(zip(frame_paths, det_results)):
        # Align input format
        d = [dict(bbox=x) for x in list(d) if x[-1] > 0.5]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        for j, item in enumerate(pose):
            kp[j, i] = item['keypoints']
        prog_bar.update()
    return kp


def gym_pose_extraction(vid, label, skip_postproc=False):
    size, frame_paths = extract_frame(vid)
    # return empty if no frame in the file
    if not frame_paths:
        return dict()
    det_results = detection_inference(args, frame_paths)
    if not skip_postproc:
        det_results = gym_det_postproc(det_results)
    pose_results = pose_inference(args, frame_paths, det_results)
    anno = dict()
    anno['keypoint'] = pose_results[..., :2]
    anno['keypoint_score'] = pose_results[..., 2]
    anno['bbox'] = det_results
    # anno['frame_dir'] = osp.splitext(osp.basename(vid))[0]
    anno['img_shape'] = size
    anno['original_shape'] = size 
    anno['total_frames'] = pose_results.shape[1]
    anno['label'] = label
    shutil.rmtree(osp.dirname(frame_paths[0]))
    return anno



def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single FineGym video')
    parser.add_argument('video', type=str, help='source video')
    parser.add_argument('anno', type=str, help='annotation location')
    parser.add_argument('output', type=str, help='output pickle name')
    
    parser.add_argument('--pre_exist', type=str,default='', help='generated skeleton file in order to avoid duplicate')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.video = global_args.video
    args.output = global_args.output
    args.anno = global_args.anno
    args.pre_exist = global_args.pre_exist
    args.skip_postproc = global_args.skip_postproc
    
    all_anno = []
    num_lines = sum(1 for line in open(args.anno))
    # slice_count = math.floor(num_lines / 10)
    fileCount = 30
    # stepper = 0

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # load pre existsing skeleton
    PES_file = []
    if args.pre_exist:
        PES = mmcv.load(args.pre_exist)
        PES_file = [x["frame_dir"] for x in PES]


    with open(args.anno) as f:
        prog_bar = mmcv.ProgressBar(num_lines)
        for line in f:
            [file, label] = line.split(" ")
            file = file + ".mp4"
            filename = join(args.video, file)
            if file in PES_file:
                print("----------------------------------")
                print("Skipping file already exist" + filename)
            elif isfile(filename):
                print("----------------------------------")
                print("Analyzing file" + filename)
                anno = gym_pose_extraction(filename, label[:-1] ,args.skip_postproc)
                anno['frame_dir'] = file
                all_anno.append(anno)
            else:
                print("===================================")
                print(filename + "Not Found")

            prog_bar.update()

            # save the result in multiple pkl files
            # if stepper >= slice_count:
            #     stepper = 0
            #     outputP = args.output + "/" + str(fileCount) + ".pkl"
            #     mmcv.dump(all_anno, outputP)
            #     fileCount += 1
            #     all_anno = []
            # stepper += 1

    outputP = args.output + "/" + "val_org_final" + ".pkl"
    final = PES + all_anno 
    mmcv.dump(final, outputP)
