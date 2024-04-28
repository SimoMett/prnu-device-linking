import sys
import os
import threading
import numpy as np
from PIL import Image

import prnu
from params import sequences
import re
from matplotlib import pyplot as plt


def main(base_dir):
    seq_idx = int(re.search(r'\d+', base_dir).group()) - 1
    seq = sequences[seq_idx]
    # removing duplicates and keeping same order
    # x = []
    # for i in seq:
    #     if i not in x:
    #         x.append(i)
    # ground_truth = prnu.gt(x, seq)
    ground_truth = prnu.gt(seq, seq)

    assert os.path.isdir(base_dir)
    images = sorted(os.listdir(base_dir), key=lambda i: int(i.removeprefix("frame").removesuffix(".png")))

    # error mitigation
    for img in images:
        assert ".png" in img
    if base_dir[-1] != '/':
        base_dir = base_dir + "/"
    ###

    def extract_func(png_path, resids, idx):
        im = Image.open(png_path)
        img_array = np.array(im)
        resids[idx] = prnu.extract_single(img_array)

    residuals = [None for _ in images]
    threads = []
    for img in images:
        thr = threading.Thread(target=extract_func, args=(base_dir + img, residuals, images.index(img)))
        thr.start()
        threads.append(thr)

    for t in threads:
        t.join()

    aligned_ncc = prnu.aligned_cc(np.array(residuals), np.array(residuals))['ncc']
    # In this case aligned_ncc is a triangular matrix. FIXME possible optimization?
    stats_cc = prnu.stats(aligned_ncc, ground_truth)
    plot = False
    if plot:
        plt.title(base_dir)
        fpr = stats_cc['fpr']
        tpr = stats_cc['tpr']
        plt.plot([i for i in range(len(fpr))], fpr, label="FPR")
        plt.plot([i for i in range(len(tpr))], tpr, label="TPR")
        plt.plot([], [], ' ', label="AUC: {:0.3f}".format(stats_cc['auc']))
        plt.grid()
        plt.legend()
        plt.show()
        # plt.savefig("test.svg")
    else:
        print(base_dir+" - area under curve (auc):", "{:0.3f}".format(stats_cc['auc']))
    return


if __name__ == "__main__":
    for i in range(1, len(sys.argv)):
        main(sys.argv[i])
