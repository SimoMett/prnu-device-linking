import re
import sys
import threading

import cv2
import numpy as np

import params
import prnu
from extract_frames import sequence_from_groundtruth, extract_frames


def main(video_path):
    mp4file = cv2.VideoCapture(video_path)
    seq = sequence_from_groundtruth(mp4file)

    # ground truth
    seq_idx = int(re.search(r'\d+', video_path.split("/")[-1]).group()) - 1
    clips_seq = params.sequences[seq_idx]
    # x = []
    # for i in clips_seq:
    #     if i not in x:
    #         x.append(i)
    # ground_truth = prnu.gt(x, clips_seq)
    ground_truth = prnu.gt(clips_seq, clips_seq)

    # fingerprint
    clips_fingerprints_k = []
    k_frames = []
    frames_count = 4  # 60
    for i in seq:
        y = list(range(i, i + frames_count))
        k_frames.append(extract_frames(mp4file, y))
    for i, f in enumerate(k_frames):
        clips_fingerprints_k.append(prnu.extract_multiple_aligned(f))

    # extract residuals from samples
    samples = extract_frames(mp4file, seq)
    residuals_w = [None for _ in samples]

    def extract_func(img_array, dest_array, idx):
        dest_array[idx] = prnu.extract_single(img_array)

    threads = []
    for i, img in enumerate(samples):
        thr = threading.Thread(target=extract_func, args=(img, residuals_w, i))
        thr.start()
        threads.append(thr)
    for t in threads:
        t.join()

    # aligned_ncc = get_and_save_aligned_ncc(w, base_dir)
    aligned_ncc = prnu.aligned_cc(np.array(clips_fingerprints_k), np.array(residuals_w))['ncc']
    stats_cc = prnu.stats(aligned_ncc, ground_truth)
    return


if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        main(sys.argv[i])
