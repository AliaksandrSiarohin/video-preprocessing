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
from util import bb_intersection_over_union, join, scheduler, crop_bbox_from_frames, save

warnings.filterwarnings("ignore")

TEST_PERSONS = {'133', '492', '174', '105', '445', '166', '525', '162', '447', '336', '071', '414', '116', 
	        '148', '502', '225', '205', '093', '141', '004', '456', '263', '418', '483', '265', '450', '201', 
	        '304', '505', '536', '510', '172', '112', '400', '270', '215', '155', '553', '343', '176', '213'}

REF_FPS = 50

def extract_bbox(frame, fa):
    bbox = fa.face_detector.detect_from_image(frame[..., ::-1])[0]    
    return bbox


def store(frame_list, tube_bbox, video_id, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=0,
                                            image_shape=args.image_shape, min_size=0, 
                                            increase_area=args.increase)
    if out is None:
        return []

    name = video_id
    person_id = video_id.split('_')[0]
    partition = 'test' if person_id in TEST_PERSONS else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format)
    return [{'bbox': '-'.join(map(str, final_bbox)), 'start': 0, 'end': len(frame_list), 'fps': REF_FPS,
             'video_id': video_id, 'height': frame_list[0].shape[0], 'width': frame_list[0].shape[1], 'partition': partition}]    

def process_video(video_id, args):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    video_path = os.path.join(args.in_folder, video_id)
    reader = imageio.get_reader(video_path)
    tube_bbox = None
    frame_list = []
    for i, frame in enumerate(reader):
        if i == 0:
            bbox = extract_bbox(resize(frame, (360, 640), preserve_range=True), fa)
            bbox = bbox * 3
            #left, top, right, bot, _ = bbox
            tube_bbox = bbox[:-1]
        frame_list.append(frame)
    return store(frame_list, tube_bbox, video_id, args)

def run(params):
    video_id, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    return process_video(video_id, args)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--in_folder", default = 'nemo-videos')
    parser.add_argument("--out_folder", default = 'nemo-256')
    parser.add_argument("--increase", default = 0.1, type=float, help='Increase bbox by this amount') 
    parser.add_argument("--format", default='.png', help='Store format (.png, .mp4)')
    parser.add_argument("--chunks_metadata", default='nemo-metadata.csv', help='Path to store metadata')
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")

    parser.add_argument("--workers", default=1, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0", help="Names of the devices comma separated.")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
        os.makedirs(args.out_folder + '/train')
        os.makedirs(args.out_folder + '/test')


    ids = sorted(os.listdir(args.in_folder))
    scheduler(ids, run, args)
