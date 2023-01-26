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
import math
import functools 

import cv2
import mmcv
import numpy as np

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

try:
    from mmtrack.apis import inference_mot, init_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_mot` and '
                      '`init_model` form `mmtracking.apis`. These apis are '
                      'required in this script! ')

mmdet_root = ''
mmpose_root = ''
mmtracking_root =''

args = abc.abstractproperty()
args.det_config = f'~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = './faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.det_score_thr = 0.5
args.pose_config = f'~/mmpose/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/3xrsn50_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = './3xrsn50_coco_256x192-58f57a68_20201127.pth'  # noqa: E501

args.tracking_config = f'~/mmtracking/configs/mot/ocsort/ocsort_faster_x_crowdhuman_mot17-private-half.py'
args.tracking_checkpoint = './ocsort_yolox_x_crowdhuman_mot17-private-half_20220813_101618-fe150582.pth' 

# alternative tracking Model
# args.tracking_config = f'~/mmtracking/configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py' 
# args.tracking_checkpoint = "./bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"


def gen_id(size=8):
    chars = string.ascii_uppercase + string.digits
    return ''.join(rd.choice(chars) for _ in range(size))


def extract_frame(video_path):
    dname = "./tmp/" + gen_id()
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

def tracking_inference(args, frame_paths):
    model = init_model(args.tracking_config, args.tracking_checkpoint, args.device)

    results = []
    print('Performing Human Tracking for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for idx,frame_path in enumerate(frame_paths):
        result = inference_mot(model, frame_path, idx)
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

def tracklet_area(tracklet):
    return sum([area(t[1]) for t in tracklet]) / len(tracklet)

def bbox_movement_distance(b1, b2):
    center_b1 = [(b1[1][0] + b1[1][2]) / 2, (b1[1][1] + b1[1][3]) / 2 ]
    center_b2 = [(b2[1][0] + b2[1][2]) / 2, (b2[1][1] + b2[1][3]) / 2 ]
    return math.hypot(center_b2[0] - center_b1[0], center_b2[1] - center_b1[1])

def tracklet_avg_movment(tracklet):
    sum = 0
    for i in range(1, len(tracklet)):
        sum += bbox_movement_distance(tracklet[i-1], tracklet[i])
    return sum / (len(tracklet) -1)

def tracklet_combine_test(t1, t2):
    # movement test
    allowance_ratio = 1.5
    t1_avg =  tracklet_avg_movment(t1)
    frame_gap = t2[0][0] - t1[-1][0]
    dist_gap = math.hypot(t2[0][1][0] - t1[-1][1][0], t2[0][1][1] - t1[-1][1][1])
    movementTest = dist_gap < frame_gap*t1_avg*allowance_ratio 

    # area test
    t1_avg = tracklet_area(t1)
    t2_avg = tracklet_area(t2)
    area_test = t1_avg * 0.5 < t2_avg and t1_avg * 2 > t2_avg

    return movementTest and area_test

def combine_crumble_tracklets(trackletsList):
    
    # connect crumble tracklets
    i = 0
    while i < len(trackletsList):
        j = i + 1
        while j < len(trackletsList):
            # j must be after i happen
            cond1 = trackletsList[i][-1][0] < trackletsList[j][0][0]
            
            # the time gap need to less than 10 frames
            cond2 = trackletsList[j][0][0] - trackletsList[i][-1][0] <= 10

            # the current tracklet  must be longer than 1
            cond3 = len(trackletsList[i]) > 1 

            if cond1 and cond2 and cond3:
                if tracklet_combine_test(trackletsList[i], trackletsList[j]):
                    trackletsList[i] = trackletsList[i] + trackletsList[j]
                    del trackletsList[j]
                    return combine_crumble_tracklets(trackletsList)

            j += 1
        i += 1

    return trackletsList
    
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

def buildtracklet(track_res):
    tracklets = defaultdict(list)
    for idx,t in enumerate(track_res):
        track_bboxes = t["track_bboxes"]
        # pdb.set_trace()
        for bbox in track_bboxes[0]:
            key = int(bbox[0])
            tracklets[key].append([idx, bbox[1:]])
    return tracklets


def bbox2tracklet(bbox):
    iou_thre = 0.3
    tracklet_id = -1
    tracklet_st_frame = {}
    tracklets = defaultdict(list)
    for t, box in enumerate(bbox):
        for idx in range(box.shape[0]):
            matched = False
            for tlet_id in range(tracklet_id, -1, -1):
                cond1 = iou(tracklets[tlet_id][-1][-1], box[idx]) >= iou_thre
                cond2 = (
                    t - tracklet_st_frame[tlet_id] - len(tracklets[tlet_id]) <
                    10)
                cond3 = tracklets[tlet_id][-1][0] != t
                if cond1 and cond2 and cond3:
                    matched = True
                    tracklets[tlet_id].append((t, box[idx]))
                    break
            if not matched:
                tracklet_id += 1
                tracklet_st_frame[tracklet_id] = t
                tracklets[tracklet_id].append((t, box[idx]))
    return tracklets

def tracklet2bbox(tracklets, totoal_frames):
    bboxes = [[] for x in range(totoal_frames)]
    for track in  tracklets:
        for t in track:
            bboxes[t[0]].append(t[1])
    return bboxes


def clean_tracklets(tracklets, totoal_frames):

    # remove empyt tracklets
    trackletsList = [v for k, v in tracklets.items() if v]

    trackletsList = combine_crumble_tracklets(trackletsList)
    
    threshold_frames = 0.5
    threshold_area = 0.2
    
    # filter out tracklet contains less than threshold_frames 
    long_clip = []
    for v in trackletsList:
        if len(v) > threshold_frames * totoal_frames:
            long_clip.append(v)
    
    avg_tracklets_area = sum([ tracklet_area(v) for v in long_clip])/ len(long_clip)
    # filter out tiny tracklets
    cleaned = []
    cleaned_area = []
    for v in long_clip:
        tarea = tracklet_area(v)
        if tarea > threshold_area * avg_tracklets_area:
            cleaned.append(v)
            cleaned_area.append(tarea)
    # pdb.set_trace()
    return cleaned, cleaned_area

def patch_tracklets2bboxes(tracklets,total_frames):

    def patch_track(tracklet):
        padding_index = 0
        # initilize time axis
        patched = [None] * total_frames
        for t in tracklet:
            time = t[0]
            patched[time] = t[1]

        # using adjancent value fill up missing frames
        first_not_none = next(x for x in patched if x is not None)
        patched[0] = first_not_none if patched[0] is None else patched[0]

        for i in range(len(patched)):
            if patched[i] is None:
                # padding the prev bboxes
                padding = [patched[i-1][0] * (1-padding_index), patched[i-1][1] * (1-padding_index),
                patched[i-1][2] * (1+padding_index), patched[i-1][3] * (1+padding_index), patched[i-1][4]]
                patched[i] = padding
        return patched
    
    patched_tracklets = [patch_track(t) for t in tracklets]
    result = []
    for t in range(total_frames):
        bboxes = []
        for tracklet in patched_tracklets:
            bboxes.append(tracklet[t])
        result.append(bboxes)
    
    return np.asarray(result)




def gym_track_refine(track_res, total_frames):

    # # 1. build up tracklets
    # tracklets = buildtracklet(track_res)

    # # 2. filter tracklets
    # tracklets = clean_tracklets(tracklets, total_frames)

    # # 3. tracklet to bboxes
    # bboxes = tracklet2bbox(tracklets)
    # pdb.set_trace()

    # return bboxes
    
    # 1. remove duplicates
    track_res = [removedup(x) for x in track_res] 
    
    # 2. generate tracklets out of track_res
    tracklets = bbox2tracklet(track_res)   

    # 3. filter out small empty and short tracklet  
    tracklets, avg_areas = clean_tracklets(tracklets, total_frames)

    # 4. patch missing frames and convert to bboxes
    return patch_tracklets2bboxes(tracklets, total_frames), avg_areas



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

def average_joint_movement(skeleton, avg_area):
    acc_frame = 0
    for frame in skeleton:
        acc_distance = 0
        for i in range(1,len(frame)):
            acc_distance += math.hypot(frame[i][0] - frame[i-1][0], frame[i][1] - frame[i-1][1])
        acc_frame += acc_distance 
    return acc_frame / avg_area


def pose_refine(skeletons, avg_areas):
    avg_distance = []
    for skeleton, avg_area in zip(skeletons, avg_areas) :
        avg_distance.append(average_joint_movement(skeleton, avg_area))
    index = avg_distance.index(max(avg_distance))
    pdb.set_trace()
    return np.asarray([skeletons[index]])



def gym_pose_extraction(vid, label, skip_postproc=False):
    size, frame_paths = extract_frame(vid)
    
    # return empty if no frame in the file
    if not frame_paths:
        return dict()

    det_results = detection_inference(args, frame_paths)

    # using mmtracking 
    # det_results = tracking_inference(args, frame_paths)

    det_results, avg_areas = gym_track_refine(det_results, len(det_results))

    pose_results = pose_inference(args, frame_paths, det_results)

    # pose_results = pose_refine(pose_results, avg_areas)
    

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
    parser.add_argument('output', type=str, help='output pickle name')

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--skip-postproc', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global_args = parse_args()
    args.device = global_args.device
    args.video = global_args.video
    args.output = global_args.output
    args.skip_postproc = global_args.skip_postproc

    all_anno = []

    debugF = "/home/haoran/Video-Swin-Transformer/data/gym/subactions/cWztehyIFkg_E_003879_003921_A_0013_0016.mp4"
    # debugF = "/home/haoran/Video-Swin-Transformer/data/gym/subactions/AZ4wWG6Rcak_E_007152_007237_A_0033_0033.mp4"
    # debugF = "/home/haoran/Video-Swin-Transformer/data/gym/subactions/e3EsDlpNo0c_E_001952_001988_A_0018_0020.mp4"
    # debugF = "/home/haoran/Video-Swin-Transformer/data/gym/subactions/1Fdwuy2V9EY_E_002942_002978_A_0028_0030.mp4"
    anno = gym_pose_extraction(debugF, 1 , args.skip_postproc)
    anno['frame_dir'] = "cWztehyIFkg_E_003879_003921_A_0013_0016.mp4"
    all_anno.append(anno)


    mmcv.dump(all_anno, args.output)
