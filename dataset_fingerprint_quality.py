import glob
from prnu_fingerprint_quality import procedure


def save_results(file, pce: list):
    with open("Dataset/dataset_dstab_tests.txt", "w") as f:
        f.write(",".join([file]+pce))


if __name__ == "__main__":
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L*/S*/*.mp4"):
        save_results(s, procedure(s))
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L*/S*/*.MOV"):
        save_results(s, procedure(s))
