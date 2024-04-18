import numpy as np
import cv2
from matplotlib import pyplot as plt
import low_pass_filter


def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)


plot.i = 0


def extract_frame(video_file, frame_number):
    mp4file = cv2.VideoCapture(video_file)
    frames_count = int(mp4file.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frames_count)

    mp4file.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = mp4file.read()
    return res, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)


def main():
    frames = []
    for i in range(3):
        res, frame = extract_frame("Dataset/D34_Google_Pixel5/Nat/jpeg-h264/L5/S4/D34_L5S4C4.mp4", (i + 1) * 300)
        frames.append(low_pass_filter.apply_lp_filter_grayscale(frame, 16))

    plot(frames[0], "frame0")
    plot(frames[1], "frame1")
    res, frames[2] = extract_frame("Dataset/D40_Motorola_MotoG9Plus/Nat/jpeg-h264/L5/S4/D40_L5S4C4.mp4", 34)
    frames[2] = low_pass_filter.apply_lp_filter_grayscale(frames[2], 16)
    plot(frames[2], "frame2")
    tt = (frames[2] - frames[1]) ** 2
    plot(tt, "frame")
    print(np.linalg.norm(tt))
    plt.show()


if __name__ == "__main__":
    main()
