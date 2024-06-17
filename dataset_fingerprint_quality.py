import glob
from prnu_fingerprint_quality import procedure


def save_results(file, pce: list):
    with open("Dataset/dataset_dstab_tests.csv", "a") as f:
        f.write(",".join([file]+[str(p) for p in pce])+"\n")


if __name__ == "__main__":
    base_str = "Dataset/D*/Nat/jpeg-h264/L5/S2/*"
    for s in glob.glob(base_str+".mp4"):
        save_results(s, procedure(s))
    for s in glob.glob(base_str+".MOV"):
        save_results(s, procedure(s))
    for s in glob.glob(base_str+".3gp"):
        save_results(s, procedure(s))
