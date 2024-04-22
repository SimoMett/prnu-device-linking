import numpy as np
import cv2
import scipy.signal
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

    mp4file.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    res, frame = mp4file.read()
    return res, cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int)


def pick_frames(video: cv2.VideoCapture, interval):
    frames_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(frames_count, "at", int(video.get(cv2.CAP_PROP_FPS)), "fps")

    picked_frames = []
    for i in range(0, frames_count, interval):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        frame = video.read()[1]
        picked_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(int))
    return picked_frames


def stability_measure(frame1, frame2, lpf_cutoff=16):
    if lpf_cutoff == 0:
        filtered_frame1 = frame1
        filtered_frame2 = frame2
    else:
        filtered_frame1 = low_pass_filter.apply_lp_filter_grayscale(frame1, lpf_cutoff)
        filtered_frame2 = low_pass_filter.apply_lp_filter_grayscale(frame2, lpf_cutoff)
    return np.linalg.norm((filtered_frame1 - filtered_frame2) ** 2)


def find_peaks(values, threshold):  # FIXME not always working
    peaks = []
    for i in range(len(values)):
        if values[i] > threshold:
            peaks.append(i)
    return peaks


def main():
    lp_cutoff = 8  # using 16 for 1080p frames. This parameter has to be adjusted for other resolutions
    # using 0 for no LP filter

    frames = pick_frames(cv2.VideoCapture("output/Seq1_Clip_L05S04.mp4"), 3)
    values = []
    for i in range(len(frames) - 1):
        values.append(stability_measure(frames[i], frames[i + 1], lp_cutoff))
    plt.subplot(2, 1, 1)
    plt.title("Seq1_Clip_L05S04.mp4")
    plt.plot([i for i in range(len(values))], np.log(values), color="blue")
    # print(find_peaks(values, np.std(values) * 3))
    # print(scipy.signal.find_peaks(np.log(values), prominence=1))
    plt.grid()

    plt.show()
    return


if __name__ == "__main__":
    main()
