import matplotlib

matplotlib.use('Agg')

from maskrcnn_benchmark.config import cfg
from maskrcnn import COCODemo

import imageio
import tqdm
import numpy as np

from argparse import ArgumentParser
from util import crop_bbox_from_frames, bb_intersection_over_union, join, compute_aspect_preserved_bbox
import os

import cv2
from skimage.transform import resize
from skimage.color import rgb2gray

from util import scheduler


def check_full_person(kps):
    head_present = np.sum((kps[:5] > args.kp_confidence_th).numpy())
    leg_present = np.sum((kps[-2:] > args.kp_confidence_th).numpy())
    return head_present and leg_present


def check_camera_motion(current_frame, previous_frame):
    flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.quantile(mag, [0.25, 0.5, 0.75], overwrite_input=True)


def store(video_path, trajectories, end, args, chunks_data, fps):
    for i, (initial_bbox, tube_bbox, start, frame_list) in enumerate(trajectories):
        out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=args.min_frames, image_shape=args.image_shape,
                                    min_size=args.min_size, increase_area=args.increase, max_pad=args.max_pad)
        if len(chunks_data) > args.max_crops:
            return
        if out is None:
            continue
        video_id = os.path.basename(video_path).split('.')[0]
        name = (video_id + "#" + str(len(chunks_data)).zfill(3) + "#"
                + str(start).zfill(6) + "#" + str(end).zfill(6) + ".mp4")
        imageio.mimsave(os.path.join(args.out_folder, name), out, fps=25)
        chunks_data.append({'bbox': '-'.join(map(str, final_bbox)), 'start': start, 'end': end, 'fps': fps,
                            'video_id': video_id, 'height': frame_list[0].shape[0], 'width': frame_list[0].shape[1]})


def process_video(video_path, detector, args):
    video = imageio.get_reader(video_path)
    fps = video.get_meta_data()['fps']
    anotations = {"keypoints": [],
                  "scores": [],
                  "keypoint_scores": [],
                  "bbox": [],
                  "flow_quantiles": [],
                  "indexes": []}

    trajectories = []
    previous_frame = None
    chunks_data = []
    try:
        for i, frame in enumerate(video):
            if args.minimal_video_size > min(frame.shape[0], frame.shape[1]):
                return chunks_data
            if i % args.sample_rate != 0:
                continue
            predictions = detector.compute_prediction(frame[:, :, ::-1])
            keypoints = predictions.get_field('keypoints').keypoints
            scores = predictions.get_field("scores")
            keypoint_scores = predictions.get_field('keypoints').get_field("logits")
            bboxes = predictions.bbox[:, :4]

            anotations['keypoints'].append(keypoints.numpy())
            anotations['scores'].append(scores.numpy())
            anotations['keypoint_scores'].append(keypoint_scores.numpy())
            anotations['bbox'].append(bboxes.numpy())
            anotations['indexes'].append(i)

            ## Check if valid person in bbox

            height_criterion = ((bboxes[:, 3] - bboxes[:, 1]) > args.mimial_person_size * frame.shape[1]).numpy()
            score_criterion = (scores > args.bbox_confidence_th).numpy()
            full_person_criterion = np.array([check_full_person(kps) for kps in keypoint_scores])

            criterion = np.logical_and(height_criterion, score_criterion)
            bboxes_distractor = bboxes.numpy()[criterion]
            criterion = np.logical_and(full_person_criterion, criterion)
            bboxes_valid = bboxes.numpy()[criterion]

            ### Check if frame is valid
            if previous_frame is None:
                previous_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))
                current_frame = previous_frame
            else:
                current_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))
            flow_quantiles = check_camera_motion(current_frame, previous_frame)
            anotations['flow_quantiles'].append(flow_quantiles[np.newaxis])
            camera_criterion = flow_quantiles[1] > args.camera_change_threshold
            previous_frame = current_frame

            no_person_criterion = len(bboxes) < 0

            criterion = no_person_criterion or camera_criterion

            if criterion:
                store(video_path, trajectories, i, args, chunks_data, fps)
                trajectories = []

            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                tube_bbox = compute_aspect_preserved_bbox(trajectory[1], args.increase)
                number_of_intersections = 0
                current_bbox = None
                for bbox in bboxes_valid:
                    intersect = bb_intersection_over_union(tube_bbox, bbox) > 0
                    if intersect:
                        current_bbox = bbox

                for bbox in bboxes_distractor:
                    intersect = bb_intersection_over_union(tube_bbox, bbox) > 0
                    if intersect:
                        number_of_intersections += 1

                if current_bbox is None:
                    not_valid_trajectories.append(trajectory)
                    continue

                if number_of_intersections > 1:
                    not_valid_trajectories.append(trajectory)
                    continue

                if bb_intersection_over_union(trajectory[0], current_bbox) < args.iou_with_initial:
                    not_valid_trajectories.append(trajectory)
                    continue

                if len(trajectory[3]) >= args.max_frames:
                    not_valid_trajectories.append(trajectory)
                    continue

                valid_trajectories.append(trajectory)

            store(video_path, not_valid_trajectories, i, args, chunks_data, fps)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes_valid:
                intersect = False
                for trajectory in trajectories:
                    tube_bbox = trajectory[1]
                    intersect = bb_intersection_over_union(tube_bbox, bbox) > 0
                    if intersect:
                        trajectory[1] = join(tube_bbox, bbox)
                        trajectory[3].append(frame)
                        break

                ## Create new trajectory
                if not intersect:
                    trajectories.append([bbox, bbox, i, [frame]])

            if len(chunks_data) > args.max_crops:
                break

    except IndexError:
        None

    anotations = {k: np.array(v) for k, v in anotations.items()}
    np.savez(os.path.join(args.annotation_folder, os.path.basename(video_path)[:-3] + 'npz'), **anotations)
    store(video_path, trajectories, i + 1, args, chunks_data, fps)
    return chunks_data

def run(params):
    video_file, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    # update the config options with the config file
    cfg.merge_from_file(args.maskrcnn_config)
    # manual override some options
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])
    detector = COCODemo(cfg, min_image_size=256, confidence_threshold=0.7)
    return process_video(os.path.join(args.video_folder, video_file), detector, args)


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--video_folder", help='Path to folder with videos')
    parser.add_argument("--device_ids", default="0,1", type=str, help="Device to run video on")
    parser.add_argument("--workers", default=1, type=int, help="Number of workers")

    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0, type=float, help='Increase bbox by this amount')
    parser.add_argument("--min_frames", default=32, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=128, type=int, help='Maximal number of frames')
    parser.add_argument("--min_size", default=256, type=int, help='Minimal allowed size')
    parser.add_argument("--max_pad", default=0, type=int, help='Minimal allowed padding')


    parser.add_argument("--out_folder", default="taichi-256", help="Folder with output videos")
    parser.add_argument("--annotation_folder", default="taichi-annotations", help="Folder for annotations")

    parser.add_argument("--bbox_confidence_th", default=0.7, type=float, help="Maskrcnn confidence for bbox")
    parser.add_argument("--kp_confidence_th", default=2, type=float, help="Maskrcnn confidence for keypoint")
    parser.add_argument("--maskrcnn_config",
                        default="/home/gin/maskrcnn-benchmark/configs/caffe2/e2e_keypoint_rcnn_R_50_FPN_1x_caffe2.yaml",
                        help="Path to keypoints config for mask rcnn")

    parser.add_argument("--mimial_person_size", default=0.10, type=float,
                        help="Minimal person size, e.g 10% of height")

    parser.add_argument("--minimal_video_size", default=300, type=int, help="Minimal size of the video")

    parser.add_argument("--camera_change_threshold", type=float, default=1)
    parser.add_argument("--sample_rate", type=int, default=5, help="Sample video rate")
    parser.add_argument("--max_crops", type=int, default=20, help="Maximal number of crops per video.")
    parser.add_argument("--chunks_metadata", default='taichi-metadata.csv', help="File to store metadata for taichi.")

    args = parser.parse_args()

    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    if not os.path.exists(args.annotation_folder):
        os.makedirs(args.annotation_folder)

    scheduler(os.listdir(args.video_folder), run, args)
