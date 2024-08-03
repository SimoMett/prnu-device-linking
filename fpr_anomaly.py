import os.path
from glob import glob
from functools import lru_cache
import cv2

from extract_frames import extract_frames
from prnu import extract_multiple_aligned, pce, crosscorr_2d
from prnu_extract_fingerprints import save_as_pickle


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
        if not os.path.exists(t+"_250.fgp"):
            save_as_pickle(t + "_250.fgp", mp4_extract_multiple_aligned(t, 0, 250, 16))


if __name__ == "__main__":
    k = mp4_extract_multiple_aligned(
        "Hybrid Dataset/D02_Apple_iPhone4_VISION/videos/outdoor/D09_V_outdoor_move_0001.mov", 0, 250, 16)
    for t in glob("Hybrid Dataset/D02_Apple_iPhone4_VISION/videos/outdoor/*.mov"):
        print(",".join("D18_Samsung_GalaxyS6/Nat/jpeg-h264/L2/S1/D09_V_outdoor_move_0001.mov",
                       t.removeprefix("Hybrid Dataset/"),
                       pce(crosscorr_2d(mp4_extract_multiple_aligned(t, 0, 250, 16), k[1]))['pce']))
