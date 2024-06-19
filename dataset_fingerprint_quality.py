import glob
from prnu_fingerprint_quality import procedure


def save_results(file, pce: list):
    with open("Dataset/dataset_dstab_tests.csv", "a") as f:
        f.write(",".join([file]+[str(p) for p in pce])+"\n")


if __name__ == "__main__":
    good_devices = [38, 16, 5, 27, 29, 9, 4, 18, 3, 28, 45, 6, 8]
    all_devices = list(range(1,47))
    format_base_str = "Dataset/D{:02d}*/Nat/jpeg-h264/L5/S2/*"
    for d in [dev for dev in all_devices if dev not in good_devices]:
        base_str = format_base_str.format(d)
        for s in glob.glob(base_str+".mp4"):
            save_results(s, procedure(s))
        for s in glob.glob(base_str+".MOV"):
            save_results(s, procedure(s))
        for s in glob.glob(base_str+".3gp"):
            save_results(s, procedure(s))
