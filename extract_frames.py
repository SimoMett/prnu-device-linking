import sys
import cv2
from PIL import Image


def extract_frame(video_capture, frame_number):
    # cv2.VideoCapture(video_file)
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

    offset = 10
    return [i for i in range(offset, tot_frames, fps * seconds_between_clips)]


def extract_frames(video_cap, sequence):
    return [extract_frame(video_cap, f)[1] for f in sequence]


if __name__ == "__main__":
    video_file = sys.argv[1]
    print(video_file)
    mp4file = cv2.VideoCapture(video_file)
    seq = sequence_from_groundtruth(mp4file)
    print(seq)
    frames = extract_frames(mp4file, seq)
    assert len(frames) == 10
    for i in range(len(frames)):
        origin_name = video_file.split('/')[-1].replace(".mp4", "_Frame")
        filename = origin_name + str(seq[i]) + ".png"
        # print(filename)
        # continue # decomment for dry run
        png_image = Image.fromarray(frames[i])
        png_image.save(filename)
