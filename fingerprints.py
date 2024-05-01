import sys
import cv2
import prnu
from extract_frames import extract_frames


def main(video_path):
    mp4file = cv2.VideoCapture(video_path)
    fps = int(mp4file.get(cv2.CAP_PROP_FPS))
    print("Fps:", fps)
    frames = extract_frames(mp4file, [i for i in range(1, fps*8, 2)])
    k = prnu.extract_multiple_aligned(frames)
    print(len(frames))
    return


if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        main(sys.argv[i])
