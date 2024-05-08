import re
import sys
import os
import threading
from multiprocessing import Pool

import cv2
import numpy as np
import pickle as pk
import params
import prnu
from extract_frames import sequence_from_groundtruth, extract_frames


def save_as_pickle(filename: str, object):
    with open(filename, "wb") as output_file:
        pk.dump(object, output_file)
    print("Generated", filename)
    return


def save_results(video_path, aligned_cc, stats_cc, pce_rot, stats_pce):
    output_path = video_path.replace(".mp4", "/")
    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "aligned_cc.csv", "w") as output_file:
        for row in aligned_cc:
            output_file.write(",".join((str(i) for i in row)) + "\n")

    with open(output_path + "pce.csv", "w") as output_file:
        for row in pce_rot:
            output_file.write(",".join((str(i) for i in row)) + "\n")

    with open(output_path + "stats_cc.csv", "w") as output_file:
        output_file.write("TPR:," + ",".join((str(i) for i in stats_cc['tpr'])) + "\n")
        output_file.write("FPR:," + ",".join((str(i) for i in stats_cc['fpr'])) + "\n")
        output_file.write("TH:," + ",".join((str(i) for i in stats_cc['th'])) + "\n")
        output_file.write("AUC:," + str(stats_cc['auc']) + "\n")
        output_file.write("EER:," + str(stats_cc['eer']) + "\n")

    with open(output_path + "stats_pce.csv", "w") as output_file:
        output_file.write("TPR:,"+",".join((str(i) for i in stats_pce['tpr'])) + "\n")
        output_file.write("FPR:,"+",".join((str(i) for i in stats_pce['fpr'])) + "\n")
        output_file.write("TH:,"+",".join((str(i) for i in stats_pce['th'])) + "\n")
        output_file.write("AUC:," + str(stats_pce['auc']) + "\n")
        output_file.write("EER:," + str(stats_pce['eer']) + "\n")

    save_as_pickle(output_path+"full_results.pickle", (aligned_cc, stats_cc, pce_rot, stats_pce))


def main(video_path):
    if os.path.isdir(video_path.replace(".mp4", "/")):
        print("Skipping",video_path+". Results already exist.")
        return
    print(video_path)
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

    # cross-correlation (CC)
    #  extract residuals from samples
    samples = extract_frames(mp4file, seq)
    pool = Pool(os.cpu_count())
    residuals_w = pool.map(prnu.extract_single, samples)
    pool.close()
    aligned_cc = prnu.aligned_cc(np.array(clips_fingerprints_k), np.array(residuals_w))['cc']
    stats_cc = prnu.stats(aligned_cc, ground_truth)

    # peak to correlation energy (PCE)
    pce_rot = np.zeros((len(clips_fingerprints_k), len(residuals_w)))

    def extract_pce_func(k1, resids_w, dest_array, i):
        for j, k2 in enumerate(resids_w):
            cc2d = prnu.crosscorr_2d(k1, k2)
            dest_array[i, j] = prnu.pce(cc2d)['pce']

    threads = []
    # FIXME ugly AF. Number of threads depend on size of clips_fingerprints_k
    for i, fp_k in enumerate(clips_fingerprints_k):
        thr = threading.Thread(target=extract_pce_func, args=(fp_k, residuals_w, pce_rot, i))
        thr.start()
        threads.append(thr)
    for t in threads:
        t.join()

    stats_pce = prnu.stats(pce_rot, ground_truth)
    save_results(video_path, aligned_cc, stats_cc, pce_rot, stats_pce)
    return


if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        main(sys.argv[i])
