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
import time

warnings.filterwarnings("ignore")

DEVNULL = open(os.devnull, 'wb')
TEST_PERSONS = ['id' + str(i) for i in range(10270, 10310)]
REF_FRAME_SIZE = 360
REF_FPS = 25


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


def extract_bbox(frame, refbbox, fa):
    bboxes = fa.face_detector.detect_from_image(frame[..., ::-1])
    bbox = max([(bb_intersection_over_union(bbox, refbbox), tuple(bbox)) for bbox in bboxes])[1]
    return np.maximum(np.array(bbox), 0)


def store(frame_list, video_count, tube_bbox, person_id, video_id, utterance):
    if len(frame_list) < args.min_frames:
<<<<<<< HEAD
        return 0
=======
        return
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a
    left, top, right, bot, _ = tube_bbox
    width = right - left
    height = bot - top
    if width < args.min_size or height < args.min_size:
        return 1
    width_inc = args.increase + max(0, height - width) / (2 * width)
    height_inc = args.increase + max(0, wigth - height) / (2 * height)
    left = max(0, left - width_inc * width)
    top = max(0, top - height_inc * height)
    right = right + width_inc * width
    bot = bot + height_inc * height
======
        return
    left = max(0, left - args.increase * width)
    top = max(0, top - args.increase * height)
    right = right + args.increase * width
    bot = bot + args.increase * height
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a
    out = [img_as_ubyte(resize(frame[int(top):int(bot), int(left):int(right)], args.image_shape)) for frame in
           frame_list]

    partition = 'test' if person_id in TEST_PERSONS else 'train'

    imageio.mimsave(
        os.path.join(args.out_folder, partition,
                     person_id + '#' + video_id + "#" + utterance + '#' + str(video_count).zfill(5) + '.mp4'),
        out)


def save_bbox_list(video_path, bbox_list):
    f = open(os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt'), 'w')

    print("LEFT,TOP,RIGHT,BOT", file=f)
    for bbox in bbox_list:
        print("%s,%s,%s,%s" % tuple(bbox[:4]), file=f)
    f.close()


def estimate_bbox(person_id, video_id, video_path, fa):
    utterance = video_path.split('#')[1]
    utterance = os.path.join(args.annotations_folder, person_id, video_id, utterance)
    reader = imageio.get_reader(video_path)

    bbox_list = []
    d = pd.read_csv(utterance, sep='\t', skiprows=6)
    frames = d['FRAME ']

<<<<<<< HEAD
    try:
        #if len(reader) != len(frames):
        #    print("Warning len video %s, len utterance %s" % (len(reader), len(frames)))
        for i, frame in enumerate(reader):
            if i >= len(frames):
                break
            val = d.iloc[i]
            mult = frame.shape[0] / REF_FRAME_SIZE
            x, y, w, h = val['X '] * mult, val['Y '] * mult, val['W '] * mult, val['H '] * mult
            bbox = extract_bbox(frame, (x, y, x + w, y + h), fa)
            bbox_list.append(bbox)
    except RuntimeError:
        None
=======
    for i, frame in enumerate(reader):
        if i >= len(frames):
            break
        val = d.iloc[i]
        mult = frame.shape[0] / REF_FRAME_SIZE
        x, y, w, h = val['X '] * mult, val['Y '] * mult, val['W '] * mult, val['H '] * mult
        bbox = extract_bbox(frame, (x, y, x + w, y + h), fa)
        bbox_list.append(bbox)
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a

    save_bbox_list(video_path, bbox_list)


def crop_video(person_id, video_id, video_path):
    utterance = video_path.split('#')[1]
    bbox_path = os.path.join(args.bbox_folder, os.path.basename(video_path)[:-4] + '.txt')
    reader = imageio.get_reader(video_path)

    d = pd.read_csv(bbox_path)
    video_count = 0
    initial_bbox = None
    tube_bbox = None
    frame_list = []
<<<<<<< HEAD
    
    low_res = 0
    try:
        for i, frame in enumerate(reader):
            bbox = np.array(d.iloc[i])

            if initial_bbox is None:
                initial_bbox = bbox
                tube_bbox = bbox

            if bb_intersection_over_union(initial_bbox, bbox) < args.iou_with_initial or len(
                    frame_list) >= args.max_frames:
                low_res += store(frame_list, video_count, tube_bbox, person_id, video_id, utterance)
                video_count += 1
                initial_bbox = bbox
                tube_bbox = bbox
                frame_list = []
            tube_bbox = join(tube_bbox, bbox)
            frame_list.append(frame)
    except RuntimeError:
        None
=======

    for i, frame in enumerate(reader):
        bbox = np.array(d.iloc[i])

        if initial_bbox is None:
            initial_bbox = bbox
            tube_bbox = bbox

        if bb_intersection_over_union(initial_bbox, bbox) < args.iou_with_initial or len(
                frame_list) >= args.max_frames:
            store(frame_list, video_count, tube_bbox, person_id, video_id, utterance)
            video_count += 1
            initial_bbox = bbox
            tube_bbox = bbox
            frame_list = []
        tube_bbox = join(tube_bbox, bbox)
        frame_list.append(frame)
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a

    store(frame_list, video_count, tube_bbox, person_id, video_id, utterance)


def download(video_id):
    video_path = os.path.join(args.video_folder, video_id + ".mp4")
    subprocess.call([args.youtube, '-f', "''best/mp4''", '--write-auto-sub', '--write-sub',
                     '--sub-lang', 'en', '--skip-unavailable-fragments',
                     "https://www.youtube.com/watch?v=" + video_id, "--output",
                     video_path], stdout=DEVNULL, stderr=DEVNULL)
    return video_path


def split_in_utterance(person_id, video_id):
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


def main(min_id, max_id, cuda='0'):
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda
    import face_alignment
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    persons = ['id' + str(i) for i in range(min_id, max_id)]

    for i, person_id in enumerate(persons):
        print("Currrent %s (%s): from %s to %s" % (person_id, float(i) / (max_id - min_id), min_id, max_id))
        video_folder = os.path.join(args.annotations_folder, person_id)
        if not os.path.exists(video_folder):
            continue
        for video_id in os.listdir(video_folder):
            try:
                intermediate_files = []
                if args.download:
                    video_path = download(video_id)
                    intermediate_files.append(video_path)

                if args.split_in_utterance:
                    chunk_names = split_in_utterance(person_id, video_id)
                    intermediate_files += chunk_names

                if args.estimate_bbox:
                    path = os.path.join(args.chunk_folder, video_id + '*.mp4')
                    for chunk in glob.glob(path):
<<<<<<< HEAD
                        estimate_bbox(person_id, video_id, chunk, fa)

=======
                        while True:
                            try:
                                estimate_bbox(person_id, video_id, chunk, fa)
                                break
                            except RuntimeError as e:
                                if str(e).startswith('CUDA'):
                                   print ("Warning: out of memory, sleep for 1s")
                                   time.sleep(1)
                                else:
                                   print (e)
                                   break                          
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a
                if args.crop:
                    path = os.path.join(args.chunk_folder, video_id + '*.mp4')
                    for chunk in glob.glob(path):
                        crop_video(person_id, video_id, chunk)

                if args.remove_intermediate_results:
                    for file in intermediate_files:
                        if os.path.exists(file):
                            os.remove(file)

<<<<<<< HEAD
            except RuntimeError as e:
                print(e)
=======
            except Exception as e:
                print("Error: %s, skipping and continue." % str(s))
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--iou_with_initial", type=float, default=0.25, help="The minimal allowed iou with inital bbox")
    parser.add_argument("--image_shape", default=(256, 256), type=lambda x: tuple(map(int, x.split(','))),
                        help="Image shape")
    parser.add_argument("--increase", default=0.1, type=float, help='Increase bbox by this amount')
    parser.add_argument("--min_frames", default=5, type=int, help='Mimimal number of frames')
    parser.add_argument("--max_frames", default=100, type=int, help='Maximal number of frames')
<<<<<<< HEAD
    parser.add_argument("--min_size", default=200, type=int, help='Minimal allowed size')

    parser.add_argument("--annotations_folder", default='txt1', help='Path to utterance annotations')
=======
    parser.add_argument("--min_size", default=64, type=int, help='Minimal allowed size')

    parser.add_argument("--annotations_folder", default='txt', help='Path to utterance annotations')
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a

    parser.add_argument("--video_folder", default='videos', help='Path to intermediate videos')
    parser.add_argument("--chunk_folder", default='chunks', help="Path to folder with video chunks")
    parser.add_argument("--bbox_folder", default='bbox', help="Path to folder with bboxes")
    parser.add_argument("--out_folder", default='vox', help='Folder for processed dataset')

    parser.add_argument("--youtube", default='./youtube-dl', help='Command for launching youtube-dl')
    parser.add_argument("--workers", default=1, type=int, help='Number of parallel workers')
    parser.add_argument("--device_ids", default="0", type=lambda x: list(x.split(',')),
                        help="Names of the devices comma separated.")

    parser.add_argument("--no-download", dest="download", action="store_false", help="Do not download videos")
    parser.add_argument("--no-split-in-utterance", dest="split_in_utterance", action="store_false",
                        help="Do not split videos in chunks")
    parser.add_argument("--no-estimate-bbox", dest="estimate_bbox", action="store_false",
                        help="Do not estimate the bboxes")
    parser.add_argument("--no-crop", dest="crop", action="store_false", help="Do not crop the videos")

<<<<<<< HEAD
    parser.add_argument("--remove-intermediate-results", dest="remove_intermediate_results", action="store_true", help="Do not crop the videos")
=======
    parser.add_argument("--remove-intermediate-results", dest="remove_intermediate_results", action="store_true", help="Remove intermediate videos")
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a

    parser.set_defaults(download=True)
    parser.set_defaults(split_in_utterance=True)
    parser.set_defaults(crop=True)
    parser.set_defaults(estimate_bbox=True)
    parser.set_defaults(remove_intermediate_results=False)

    args = parser.parse_args()

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

    ids = sorted(os.listdir(args.annotations_folder))
    minimum, maximum = int(ids[0][2:]), int(ids[-1][2:])
<<<<<<< HEAD
    ln = (maximum - minimum) // args.workers + 1
=======
    ln = (maximum - minimum + 1) // args.workers
>>>>>>> 752bf11ad2e3cf79551ebe339e4599e8c59c522a
    device_ind = 0
    processes = []
    for begin in range(minimum, maximum + 1, ln):
        p = Process(target=main, args=(begin, begin + ln, args.device_ids[device_ind % len(args.device_ids)]))
        p.start()
        processes.append(p)
        device_ind += 1

    for p in processes:
        p.join()
