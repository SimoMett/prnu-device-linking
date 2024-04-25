import sys
import os
import glob
import cv2
from PIL import Image


def extract_frame(video_capture, frame_number):
    mp4file = video_capture

    mp4file.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = mp4file.read()
    return res, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def sequence_from_groundtruth(video_capture):
    # The ground truth is: every 10 seconds there's a clip change
    seconds_between_clips = 10
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    print("Fps:", fps)

    tot_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frames count", tot_frames)

    # This offset is needed because all the videos have 3009 frames
    #  which is weird because every video should be 100 seconds long at 30fps, i.e. 3000 frames long.
    #  Maybe some floating errors?
    # Correction: there are also videos at 25 fps and 2504 frames long
    offset = 10
    return [i for i in range(offset, tot_frames, fps * seconds_between_clips)]


def extract_frames(video_cap, sequence):
    return [extract_frame(video_cap, f)[1] for f in sequence]


def export_pngs(video_file):
    print(video_file)
    mp4file = cv2.VideoCapture(video_file)
    seq = sequence_from_groundtruth(mp4file)
    print(seq)
    frames = extract_frames(mp4file, seq)
    assert len(frames) == 10

    origin_name = video_file.split('/')[-1].replace(".mp4", "")
    os.makedirs(origin_name, exist_ok=True)
    for i in range(len(frames)):
        filename = origin_name + "/frame" + str(seq[i]) + ".png"
        # print(filename)
        # continue # decomment for dry run
        png_image = Image.fromarray(frames[i])
        png_image.save(filename)


if __name__ == "__main__":
    for mp4 in glob.glob(sys.argv[1]):
        export_pngs(mp4)
