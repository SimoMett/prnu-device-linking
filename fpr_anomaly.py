import os.path
from glob import glob
from functools import lru_cache
import cv2
import numpy as np

import roc_stats
from extract_frames import extract_frames
from generate_video_sequences import get_clips_paths
from prnu import extract_multiple_aligned, pce, crosscorr_2d
from prnu_extract_fingerprints import save_as_pickle, compute_pce


@lru_cache(maxsize=None)
def mp4_extract_multiple_aligned(video_path: str, start, end, processes):
    mp4 = cv2.VideoCapture(video_path)
    f = extract_frames(mp4, list(range(start, end)))
    K = extract_multiple_aligned(f, processes=processes, batch_size=processes)
    return K


def generate_all_fingerprints():
    for t in glob("Hybrid Dataset/*/Nat/jpeg-h264/L*/S*/.mp4") + glob(
            "Hybrid Dataset/*/Nat/jpeg-h264/L*/S*/.mov") + glob("Hybrid Dataset/*/Nat/jpeg-h264/L*/S*/.3gp") + glob(
        "Hybrid Dataset/*_VISION/videos/*/*.mov") + glob("Hybrid Dataset/*_VISION/videos/*/*.3gp"):
        if not os.path.exists(t + "_250.fgp"):
            save_as_pickle(t + "_250.fgp", mp4_extract_multiple_aligned(t, 0, 250, 8))


def save_results(output_path, row_names, pce_rot):
    with open(output_path, "w") as output_file:
        output_file.write(",".join(row_names) + "\n")
        for ri, row in enumerate(pce_rot):
            output_file.write(row_names[ri] + "," + ",".join(("{:.1f}".format(i) for i in row)) + "\n")


if __name__ == "__main__":
    devices = ["D03_Huawei_P8Lite", "D38_Xiaomi_Redmi5Plus", "D08_Lenovo_TabE7", "D05_Huawei_P9Lite",
               "D28_Motorola_MotoG", "D18_Samsung_GalaxyS6", "D02_Apple_iPhone4_VISION", "D04_Xiaomi_RedmiNote8T"]
    did = [3, 38, 8, 5, 28, 18, 4]

    for d in did[1::]:
        out = "D" + str(d) + "_fpr_anomaly_test.csv"
        tt = []
        for c in get_clips_paths(d, "Hybrid Dataset/"):
            if any(x in c for x in ["/L3", "/L2", "/L7"]):
                tt.append(c)

        fingerprints = [[t.split('/')[-1], mp4_extract_multiple_aligned(t, 0, 250, 8)] for t in tt]

        pce_rot = compute_pce([f[1] for f in fingerprints], [f[1] for f in fingerprints])
        save_results(out, [f[0] for f in fingerprints], pce_rot)
        fp, tp, fn, tn = roc_stats.get_roc_stats_by_threshold(np.identity(len(pce_rot)), pce_rot, 60)
        print(fp, tp, fn, tn)
