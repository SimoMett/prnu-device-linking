import glob
import sys
from prnu_fingerprint_quality import procedure
import argparse


def save_results(file, pce: list):
    with open("Dataset/dataset_dstab_tests.csv", "a") as f:
        f.write(",".join([file]+[str(p) for p in pce])+"\n")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--processes", type=int, nargs=1)
    args = arg_parser.parse_args(sys.argv[1:])

    good_devices = [38, 5, 27, 29, 9, 4, 18, 3, 28, 45, 6, 8]
    all_devices = list(range(1, 47))
    format_base_str = "Dataset/D{:02d}*/Nat/jpeg-h264/L*/S*/*"
    for d in good_devices:
        base_str = format_base_str.format(d)
        for s in glob.glob(base_str + ".mp4") + glob.glob(base_str + ".MOV") + glob.glob(base_str + ".3gp"):
            if args.processes is not None:
                save_results(s, procedure(s, args.processes[0]))
            else:
                save_results(s, procedure(s))
