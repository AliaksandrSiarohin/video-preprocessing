import numpy as np
import pandas as pd
import imageio
import os
import subprocess
import warnings
import glob
import time
from util import bb_intersection_over_union, join, scheduler, crop_bbox_from_frames, save
from argparse import ArgumentParser
from skimage.transform import resize
warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
REF_FRAME_SIZE = 360
REF_FPS = 25


def extract_bbox(frame, refbbox, fa):
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    if len(bboxes) != 0:
        bbox = max([(bb_intersection_over_union(bbox, refbbox), tuple(bbox)) for bbox in bboxes])[1]
    else:
        bbox = np.array([0, 0, 0, 0, 0])
    return np.maximum(np.array(bbox), 0)


def save_bbox_list(video_path, bbox_list):
    f = open(os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt'), 'w')
    print("LEFT,TOP,RIGHT,BOT", file=f)
    for bbox in bbox_list:
        print("%s,%s,%s,%s" % tuple(bbox[:4]), file=f)
    f.close()


def estimate_bbox(person_id, video_id, video_path, fa, args):
    utterance = video_path.split('#')[1]
    utterance = os.path.join(args.annotations_folder, person_id, video_id, utterance)
    reader = imageio.get_reader(video_path)
    bbox_list = []
    d = pd.read_csv(utterance, sep='\t', skiprows=6)
    frames = d['FRAME ']

    try:
        for i, frame in enumerate(reader):
            if i >= len(frames):
                break
            val = d.iloc[i]
            mult = frame.shape[0] / REF_FRAME_SIZE
            frame = resize(frame, (REF_FRAME_SIZE, int(frame.shape[1] / mult)), preserve_range=True)
 
            if args.dataset_version == 1:
                x, y, w, h = val['X '], val['Y '], val['W '], val['H ']
            else:
                x, y, w, h = val['X '] *  frame.shape[1], val['Y '] * frame.shape[0], val['W '] * frame.shape[1], val['H '] * frame.shape[0]
            bbox = extract_bbox(frame, (x, y, x + w, y + h), fa)
            bbox_list.append(bbox * mult)
    except IndexError:
        None

    save_bbox_list(video_path, bbox_list)


def store(frame_list, tube_bbox, video_id, utterance, person_id, start, end, video_count, chunk_start, args):
    out, final_bbox = crop_bbox_from_frames(frame_list, tube_bbox, min_frames=args.min_frames,
                                            image_shape=args.image_shape, min_size=args.min_size, 
                                            increase_area=args.increase)
    if out is None:
        return []

    start += round(chunk_start * REF_FPS)
    end += round(chunk_start * REF_FPS)
    name = (person_id + "#" + video_id + "#" + utterance + '#' + str(video_count).zfill(3) + ".mp4")
    partition = 'test' if person_id in TEST_PERSONS else 'train'
    save(os.path.join(args.out_folder, partition, name), out, args.format)
    return [{'bbox': '-'.join(map(str, final_bbox)), 'start': start, 'end': end, 'fps': REF_FPS,
             'video_id': '#'.join([video_id, person_id]), 'height': frame_list[0].shape[0], 
             'width': frame_list[0].shape[1], 'partition': partition}]


def crop_video(person_id, video_id, video_path, args):
    utterance = video_path.split('#')[1]
    bbox_path = os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')
    reader = imageio.get_reader(video_path)

    chunk_start = float(video_path.split('#')[2].split('-')[0])

    d = pd.read_csv(bbox_path)
    video_count = 0
    initial_bbox = None
    start = 0
    tube_bbox = None
    frame_list = []
    chunks_data = []

    try:
        for i, frame in enumerate(reader):
            bbox = np.array(d.iloc[i])

            if initial_bbox is None:
                initial_bbox = bbox
                start = i
                tube_bbox = bbox

            if bb_intersection_over_union(initial_bbox, bbox) < args.iou_with_initial or len(
                    frame_list) >= args.max_frames:
                chunks_data += store(frame_list, tube_bbox, video_id, utterance, person_id, start, i, video_count, chunk_start,
                                     args)
                video_count += 1
                initial_bbox = bbox
                start = i
                tube_bbox = bbox
                frame_list = []
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
    except IndexError as e:
        None
    
    chunks_data += store(frame_list, tube_bbox, video_id, utterance, person_id, start, i + 1, video_count, chunk_start,
                         args)

    return chunks_data


def download(video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path


def split_in_utterance(person_id, video_id, args):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")

    if not os.path.exists(video_path):
        print("No video file %s found, probably broken link" % video_id)
        return []

    utterance_folder = os.path.join(args.annotations_folder, person_id, video_id)
    utterance_files = sorted(os.listdir(utterance_folder))
    utterances = [pd.read_csv(os.path.join(utterance_folder, f), sep='\t', skiprows=6) for f in
                  utterance_files]

    chunk_names = []

    for i, utterance in enumerate(utterances):
        first_frame, last_frame = utterance['FRAME '].iloc[0], utterance['FRAME '].iloc[-1]

        first_frame = round(first_frame / float(REF_FPS), 3)
        last_frame = round(last_frame / float(REF_FPS), 3)

        chunk_name = os.path.join(args.chunk_folder,
                                  video_id + '#' + utterance_files[i] + '#' + str(first_frame) + '-' + str(
                                      last_frame) + '.mp4')

        chunk_names.append(chunk_name)

        subprocess.call(['ffmpeg', '-y', '-i', video_path, '-qscale:v',
                         '5', '-r', '25', '-threads', '1', '-ss', str(first_frame), '-to', str(last_frame),
                         '-strict', '-2', '-deinterlace', chunk_name],
                        stdout=DEVNULL, stderr=DEVNULL)
    return chunk_names


def run(params):
    person_id, device_id, args = params
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id
    # update the config options with the config file
    if args.estimate_bbox:
        import face_alignment
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    video_folder = os.path.join(args.annotations_folder, person_id)
    
    chunks_data = []
    for video_id in os.listdir(video_folder):
        intermediate_files = []
        try:
            if args.download:
                video_path = download(video_id, args)
                intermediate_files.append(video_path)

            if args.split_in_utterance:
                chunk_names = split_in_utterance(person_id, video_id, args)
                intermediate_files += chunk_names

            if args.estimate_bbox:
                path = os.path.join(args.chunk_folder, video_id + '*.mp4')
                for chunk in glob.glob(path):
                    while True:
                        try:
                            estimate_bbox(person_id, video_id, chunk, fa, args)
                            break
                        except RuntimeError as e:
                            if str(e).startswith('CUDA'):
                                print("Warning: out of memory, sleep for 1s")
                                time.sleep(1)
                            else:
                                print(e)
                                break

            if args.crop:
                path = os.path.join(args.chunk_folder, video_id + '*.mp4')
                for chunk in glob.glob(path):
                    if not os.path.exists(os.path.join(args.bbox_folder, os.path.basename(chunk)[:-4] + '.txt')):
                       print ("BBox not found %s" % chunk)
                       continue
                    chunks_data += crop_video(person_id, video_id, chunk, args)

            if args.remove_intermediate_results:
                for file in intermediate_files:
                    if os.path.exists(file):
                        os.remove(file)
        except Exception as e:
            print (e)
    return chunks_data


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--dataset_version", default=1, type=int, choices=[1, 2], help='Version of Vox celeb dataset 1 or 2')

    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--min_frames", default=64, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=1024, type=int, help='Maximal number of frames')
    parser.add_argument("--min_size", default=256, type=int, help='Minimal allowed size')
    parser.add_argument("--format", default='.png', help='Store format (.png, .mp4)')

    parser.add_argument("--annotations_folder", default='txt', help='Path to utterance annotations')

    parser.add_argument("--video_folder", default='videos', help='Path to intermediate videos')
    parser.add_argument("--chunk_folder", default='chunks', help="Path to folder with video chunks")
    parser.add_argument("--bbox_folder", default='bbox', help="Path to folder with bboxes")
    parser.add_argument("--out_folder", default='vox-png', help='Folder for processed dataset')
    parser.add_argument("--chunks_metadata", default='vox-metadata.csv', help='File with metadata')

    parser.add_argument("--youtube", default='./youtube-dl', help='Command for launching youtube-dl')
    parser.add_argument("--workers", default=1, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0", help="Names of the devices comma separated.")

    parser.add_argument("--data_range", default=(0, 10000), type=lambda x: tuple(map(int, x.split('-'))), help="Range of ids for processing")
 

    parser.add_argument("--no-download", dest="download", action="store_false", help="Do not download videos")
    parser.add_argument("--no-split-in-utterance", dest="split_in_utterance", action="store_false",
                        help="Do not split videos in chunks")
    parser.add_argument("--no-estimate-bbox", dest="estimate_bbox", action="store_false",
                        help="Do not estimate the bboxes")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Do not crop the videos")

    parser.add_argument("--remove-intermediate-results", dest="remove_intermediate_results", action="store_true",
                        help="Remove intermediate videos")

    parser.set_defaults(download=True)
    parser.set_defaults(split_in_utterance=True)
    parser.set_defaults(crop=True)
    parser.set_defaults(estimate_bbox=True)
    parser.set_defaults(remove_intermediate_results=False)

    args = parser.parse_args()

    if args.dataset_version == 1:
        TEST_PERSONS = ['id' + str(i) for i in range(10270, 10310)]
    else:
        TEST_PERSONS = ['id07874', 'id00017', 'id00081', 'id09017', 'id08374', 'id04276', 'id03862', 'id00817', 'id00154',
                        'id02317', 'id06484', 'id07312', 'id03041', 'id05124', 'id03980', 'id05459', 'id04627', 'id08548',
                        'id01333', 'id02725', 'id05999', 'id06310', 'id08149', 'id04094', 'id08392', 'id02577', 'id01460',
                        'id02057', 'id08701', 'id00812', 'id00926', 'id03839', 'id06104', 'id07426', 'id08552', 'id01567',
                        'id03382', 'id02286', 'id03347', 'id08456', 'id02745', 'id00061', 'id01066', 'id03969', 'id06913',
                        'id01228', 'id02086', 'id08911', 'id01298', 'id06811', 'id07961', 'id04536', 'id01509', 'id01892',
                        'id08696', 'id06692', 'id01593', 'id01000', 'id01618', 'id04253', 'id04657', 'id04656', 'id03030',
                        'id01437', 'id02548', 'id01106', 'id04570', 'id05176', 'id05816', 'id00562', 'id02181', 'id07802',
                        'id03978', 'id04030', 'id03789', 'id04295', 'id00866', 'id07868', 'id04119', 'id01989', 'id07414',
                        'id01041', 'id03178', 'id04232', 'id03127', 'id06209', 'id03677', 'id04006', 'id05850', 'id02576',
                        'id05594', 'id01541', 'id05055', 'id07354', 'id01224', 'id03524', 'id02445', 'id07663', 'id05015',
                        'id07494', 'id04950', 'id04478', 'id02685', 'id02542', 'id05714', 'id02465', 'id05654', 'id05202',
                        'id00419', 'id03981', 'id04366', 'id07396', 'id02019', 'id01822', 'id06816', 'id07621', 'id07620', 'id04862']

    if not os.path.exists(args.video_folder):
        os.makedirs(args.video_folder)
    if not os.path.exists(args.chunk_folder):
        os.makedirs(args.chunk_folder)
    if not os.path.exists(args.bbox_folder):
        os.makedirs(args.bbox_folder)
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)
    for partition in ['test', 'train']:
        if not os.path.exists(os.path.join(args.out_folder, partition)):
            os.makedirs(os.path.join(args.out_folder, partition))

    ids = set(os.listdir(args.annotations_folder))
    ids_range = {'id' + str(num).zfill(5) for num in range(args.data_range[0], args.data_range[1])}
    ids = sorted(list(ids.intersection(ids_range)))
    scheduler(ids, run, args)
