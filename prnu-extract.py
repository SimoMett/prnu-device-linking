import sys
import os
import numpy as np
from PIL import Image

import prnu


def main():
    base_dir = sys.argv[1]
    assert os.path.isdir(base_dir)

    images = sorted(os.listdir(base_dir), key=lambda i: int(i.removeprefix("frame").removesuffix(".png")))
    residuals = []
    for img in images:
        assert ".png" in img
    for img in images:
        im = Image.open(base_dir+img)
        img_array = np.array(im)
        residuals.append(prnu.extract_single(img_array))

    aligned_ncc = prnu.aligned_cc(residuals, residuals)['ncc']
    return


if __name__ == "__main__":
    main()
