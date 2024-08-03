import pickle
import re
from glob import glob

import cv2

import prnu
from extract_frames import extract_frames
from prnu import extract_multiple_aligned, pce, crosscorr_2d
from prnu_extract_fingerprints import devs_sequences, compute_pce, get_clip_fingerprint, save_as_pickle
from scene_detect import sequence_from_scenedetect


def mp4_extract_multiple_aligned(video_path: str, start, end, processes):
    mp4 = cv2.VideoCapture(video_path)
    f = extract_frames(mp4, list(range(start, end)))
    K = extract_multiple_aligned(f, processes=processes, batch_size=processes)
    return K


if __name__ == "__main__":
    k1 = mp4_extract_multiple_aligned("Hybrid Dataset/D03_Huawei_P8Lite/Nat/jpeg-h264/L7/S2/D03_L7S2C4.mp4", 0, 250, 16)
    for s in glob("Hybrid Dataset/D03_Huawei_P8Lite/Nat/jpeg-h264/L*/*/*.mp4"):
        print(s, pce(crosscorr_2d(k1, mp4_extract_multiple_aligned(s, 0, 250, 16)))['pce'])
