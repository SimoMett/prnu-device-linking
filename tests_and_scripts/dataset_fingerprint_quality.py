import glob
import sys
from tests_and_scripts.prnu_fingerprint_quality import procedure


if __name__ == "__main__":
    #s = sys.argv[1]
    #procedure(s)
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L5/S2/*.mp4"):
        print(s)
        procedure(s)
    for s in glob.glob("Dataset/D*/Nat/jpeg-h264/L5/S2/*.MOV"):
        print(s)
        procedure(s)
