import glob
from prnu_fingerprint_quality import procedure


if __name__ == "__main__":
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L5/S2/*.mp4"):
        procedure(s)
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L5/S2/*.MOV"):
        procedure(s)
