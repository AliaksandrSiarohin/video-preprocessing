import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize


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

def crop_bbox_from_frames(frame_list, tube_bbox, min_frames=16, image_shape=(256, 256), min_size=200,
                          increase_area=0.1):
    frame_shape = frame_list[0].shape
    # Filter short sequences
    if len(frame_list) < min_frames:
        return None
    left, top, right, bot = tube_bbox
    width = right - left
    height = bot - top
    # Filter if it is too small
    if max(width, height) < min_size:
        return None

    left, top, right, bot = compute_aspect_preserved_bbox(tube_bbox, increase_area)

    # If something is out of bounds, pad with white
    left_oob = -min(0, left)
    right_oob = right - min(right, frame_shape[1])
    top_oob = -min(0, top)
    bot_oob = bot - min(bot, frame_shape[0])

    left += left_oob
    right += left_oob
    top += top_oob
    bot += top_oob

    padded = [np.pad(frame, pad_width=((top_oob, bot_oob), (left_oob, right_oob), (0, 0)),
                     mode='constant', constant_values=255) for frame in frame_list]
    selected = [frame[top:bot, left:right] for frame in padded]
    out = [img_as_ubyte(resize(frame, image_shape, anti_aliasing=True)) for frame in selected]

    return out

from multiprocessing import Pool
from itertools import cycle
from tqdm import tqdm


def scheduler(data_list, fn, args):
    device_ids = args.device_ids.split(",")
    pool = Pool(processes=args.workers)
    args_list = cycle([args])
    for _ in tqdm(enumerate(pool.imap_unordered(fn, zip(data_list, cycle(device_ids), args_list)))):
        None
