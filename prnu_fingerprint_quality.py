import sys
import os

import cv2
from tqdm import tqdm

from extract_frames import extract_frames
from scene_detect import sequence_from_scenedetect
from prnu_extract_fingerprints import extract_and_test_multiple_aligned


def procedure(video_path: str, threads_count=os.cpu_count() - 2):
    if os.path.isdir(video_path.replace(".mp4", "/")):
        print("Skipping", video_path + ". Results already exist.")
        return

    mp4file = cv2.VideoCapture(video_path)
    fps = int(mp4file.get(cv2.CAP_PROP_FPS))
    tot_frames = int(mp4file.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_path + ": Fps:", str(fps) + ", frames count:", tot_frames)
    # seq = sequence_from_scenedetect(video_path)
    seq = [fps, tot_frames]  # removing frames of the first second

    # fingerprint
    results = []
    for i in range(len(seq) - 1):
        print("Extracting..")
        max_frames = 700
        f = extract_frames(mp4file, list(range(seq[i], seq[i + 1]))[:max_frames])
        steps = [max_frames, 400, 200, 120, 80]
        for s in steps:
            f = f[:s]
            results.insert(0, extract_and_test_multiple_aligned(f, processes=threads_count)[1])

    return [int(r) for r in results]  # rounding (I don't want a 13 digits float value)


if __name__ == "__main__":
    for s in sys.argv[1::]:
        procedure(s)
