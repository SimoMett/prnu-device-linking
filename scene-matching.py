import numpy as np
import cv2


def main():
    mp4file = cv2.VideoCapture("D34_L5S4C4.mp4")
    mp4file.set(cv2.CAP_PROP_POS_FRAMES, 100)
    res, frame = mp4file.read()
    print(res)


if __name__ == "__main__":
    main()
