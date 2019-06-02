import numpy as np
import pandas as pd
import imageio
from skimage.transform import resize
from argparse import ArgumentParser
from skimage import img_as_ubyte
import os
import subprocess
from multiprocessing import Process
import warnings
import glob
from tqdm import tqdm
import face_alignment

warnings.filterwarnings("ignore")

TEST_VIDEO = [name.replace('.png', '.mp4') for name in os.listdir('../mtm/data/nemo/test')]

def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB, None)


def extract_bbox(frame):
    bbox = fa.face_detector.detect_from_image(frame[..., ::-1])[0]    
    return bbox
    
 #np.maximum(np.array(bbox[0]), 0)


def store(frame_list, tube_bbox, video_id):
    left, top, right, bot, _ = tube_bbox
    inc_left, inc_top, inc_right, inc_bot = args.increase

    width = right - left
    height = bot - top

    left = max(0, left - inc_left * width)
    top = max(0, top -  inc_top * height)

    right = right + inc_right * width
    bot = bot + inc_bot * height

    width = right - left
    height = bot - top
    
    if height > width:
        diff = height - width
        left = max(0, left - diff / 2)
        right = right + diff / 2
    else:
        diff = width - height
        top = max(0, top - diff / 2)
        bot = bot + diff / 2
    
    width = right - left
    height = bot - top

    out = [img_as_ubyte(resize(frame[int(top):int(bot), int(left):int(right)], args.image_shape)) for frame in
           frame_list]

    partition = 'test' if video_id in TEST_VIDEO else 'train'
    
    imageio.mimsave(os.path.join(args.out_folder, partition, video_id), out)


def process_video(video_id):
    video_path = os.path.join(args.in_folder, video_id)
    reader = imageio.get_reader(video_path)
    tube_bbox = None
    frame_list = []
    for i, frame in enumerate(reader):
        if i % 4 != 0:
           continue
        bbox = extract_bbox(resize(frame, (360, 640), preserve_range=True))
        bbox = bbox * 3
        left, top, right, bot, _ = bbox

        if tube_bbox is None:
           tube_bbox = bbox
        bbox = join(tube_bbox, bbox)
        frame_list.append(frame)
    store(frame_list, tube_bbox, video_id)


if __name__ == "__main__":
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    parser = ArgumentParser()

    parser.add_argument("--in_folder", default = 'nemo-videos')
    parser.add_argument("--out_folder", default = 'nemo-256')
    parser.add_argument("--increase", default = (0.1, 0.1, 0.1, 0.1), type=lambda x: tuple(map(int, x.split(','))),) 

    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        os.makedirs(args.out_folder + '/train')
        os.makedirs(args.out_folder + '/test')

    for video_id in tqdm(os.listdir(args.in_folder)):
        print (video_id)
        process_video(video_id)
