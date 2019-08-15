import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize
import imageio

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


def one_box_inside_other(boxA, boxB):
    xA = boxA[0] <= boxB[0]
    yA = boxA[1] <= boxB[1]
    xB = boxA[2] >= boxB[2]
    yB = boxA[3] >= boxB[3]
    return xA and yA and xB and yB

def join(tube_bbox, bbox):
    xA = min(tube_bbox[0], bbox[0])
    yA = min(tube_bbox[1], bbox[1])
    xB = max(tube_bbox[2], bbox[2])
    yB = max(tube_bbox[3], bbox[3])
    return (xA, yA, xB, yB)


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)

def compute_increased_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    left = int(left - increase_area * width)
    top = int(top - increase_area * height)
    right = int(right + increase_area * width)
    bot = int(bot + increase_area * height)

    return (left, top, right, bot)



def crop_bbox_from_frames(frame_list, tube_bbox, min_frames=16, image_shape=(256, 256), min_size=200,
                          increase_area=0.1, aspect_preserving=True):
    frame_shape = frame_list[0].shape
    # Filter short sequences
    if len(frame_list) < min_frames:
        return None, None
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top
    # Filter if it is too small
    if max(width, height) < min_size:
        return None, None
    
    if aspect_preserving:
        left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)
    else:
        left, top, right, bot = compute_increased_bbox(tube_bbox, increase_area)
 
    # Compute out of bounds
    left_oob = -min(0, left)
    right_oob = right - min(right, frame_shape[1])
    top_oob = -min(0, top)
    bot_oob = bot - min(bot, frame_shape[0])
    
    #Not use near the border
    if max(left_oob / float(width), right_oob / float(width), top_oob  / float(height), bot_oob / float(height)) > 0:
        return [None, None]

    selected = [frame[top:bot, left:right] for frame in frame_list]
    if image_shape is not None:
        out = [img_as_ubyte(resize(frame, image_shape, anti_aliasing=True)) for frame in selected]
    else:
        out = selected
 
    return out, [left, top, right, bot]

from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm
import os

def scheduler(data_list, fn, args):
    device_ids = args.device_ids.split(",")
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    f = open(args.chunks_metadata, 'w')
    line = "{video_id},{start},{end},{bbox},{fps},{width},{height},{partition}"
    print (line.replace('{', '').replace('}', ''), file=f)
    for chunks_data in tqdm(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list))):
        for data in chunks_data:
            print (line.format(**data), file=f)
            f.flush()
    f.close()

def save(path, frames, format):
    if format == '.mp4':
        imageio.mimsave(path, frames)
    elif format == '.png':
        if os.path.exists(path):
            print ("Warning: skiping video %s" % os.path.basename(path))
            return
        else:
            os.makedirs(path)
        for j, frame in enumerate(frames):
            imageio.imsave(os.path.join(path, str(j).zfill(7) + '.png'), frames[j]) 
    else:
        print ("Unknown format %s" % format)
        exit()
