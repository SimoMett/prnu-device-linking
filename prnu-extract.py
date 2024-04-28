import sys
import os
import numpy as np
from PIL import Image

import prnu
from params import sequences


def main():
    seq = sequences[0]
    # removing duplicates and keeping same order
    # x = []
    # for i in seq:
    #     if i not in x:
    #         x.append(i)
    # ground_truth = prnu.gt(x, seq)
    ground_truth = prnu.gt(seq, seq)

    base_dir = sys.argv[1]
    assert os.path.isdir(base_dir)

    images = sorted(os.listdir(base_dir), key=lambda i: int(i.removeprefix("frame").removesuffix(".png")))
    residuals = []
    # error mitigation
    for img in images:
        assert ".png" in img
    if base_dir[-1] != '/':
        base_dir = base_dir + "/"
    ###
    # TODO multithreading?
    for img in images:
        im = Image.open(base_dir + img)
        img_array = np.array(im)
        residuals.append(prnu.extract_single(img_array))

    aligned_ncc = prnu.aligned_cc(np.array(residuals), np.array(residuals))['ncc']
    # In this case aligned_ncc is a triangular matrix. FIXME possible optimization?
    stats_cc = prnu.stats(aligned_ncc, ground_truth)
    # print("false-positive rate:", stats_cc['fpr']) # TODO plot?
    # print("true-positive rate:", stats_cc['tpr']) # TODO plot?
    print("area under curve (auc):", "{:0.3f}".format(stats_cc['auc']))
    return


if __name__ == "__main__":
    main()
