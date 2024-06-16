import sys
import os
from multiprocessing import Pool

import cv2
import numpy as np
from tqdm import tqdm

import prnu
from extract_frames import extract_frames
from prnu import inten_sat_compact, noise_extract_compact, inten_scale, saturation, rgb2gray, zero_mean_total, \
    wiener_dft
from scene_detect import sequence_from_scenedetect


def extract_and_test_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                                      batch_size=os.cpu_count(), tqdm_str: str = '') -> (np.ndarray, float):
    """
    @author: Luca Bondi (luca.bondi@polimi.it)
    @author: Paolo Bestagini (paolo.bestagini@polimi.it)
    @author: Nicolò Bonettini (nicolo.bonettini@polimi.it)
    Politecnico di Milano 2018
        Modified by Matteo Simonetti

    Extract PRNU from a list of images. Images are supposed to be the same size and properly oriented
    :param tqdm_str: tqdm description (see tqdm documentation)
    :param batch_size: number of parallel processed images
    :param processes: number of parallel processes
    :param imgs: list of images of size (H,W,Ch) and type np.uint8
    :param levels: number of wavelet decomposition levels
    :param sigma: estimated noise power
    :return: PRNU
    """
    assert (isinstance(imgs[0], np.ndarray))
    assert (imgs[0].ndim == 3)
    assert (imgs[0].dtype == np.uint8)

    h, w, ch = imgs[0].shape

    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)

    if processes is None or processes > 1:
        # TODO more refactoring
        # First half of imgs
        RPsum_a = np.zeros((h, w, ch), np.float32)
        NN_a = np.zeros((h, w, ch), np.float32)
        block_a = imgs[:int(len(imgs) / 2)]
        args_list = [(im, levels, sigma) for im in block_a]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(block_a)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN_a += ni
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(block_a)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum_a += wi
            del wi_list

        # Second half of imgs
        RPsum_b = np.zeros((h, w, ch), np.float32)
        NN_b = np.zeros((h, w, ch), np.float32)
        block_b = imgs[int(len(imgs) / 2):]
        args_list = [(im, levels, sigma) for im in block_b]
        pool = Pool(processes=processes)

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(block_b)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
            nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for ni in nni:
                NN_b += ni
            del nni

        for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(block_b)), disable=tqdm_str == '',
                               desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
            wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
            for wi in wi_list:
                RPsum_b += wi
            del wi_list
        pool.close()

    else:  # Single process
        # First half
        block_a = imgs[:int(len(imgs) / 2)]
        RPsum_a = np.zeros((h, w, ch), np.float32)
        NN_a = np.zeros((h, w, ch), np.float32)
        for im in tqdm(block_a, disable=tqdm_str is None, desc="1/2" + tqdm_str, dynamic_ncols=True):
            RPsum_a += noise_extract_compact((im, levels, sigma))
            NN_a += (inten_scale(im) * saturation(im)) ** 2

        # Second half
        block_b = imgs[int(len(imgs) / 2):]
        RPsum_b = np.zeros((h, w, ch), np.float32)
        NN_b = np.zeros((h, w, ch), np.float32)
        for im in tqdm(block_b, disable=tqdm_str is None, desc="2/2" + tqdm_str, dynamic_ncols=True):
            RPsum_b += noise_extract_compact((im, levels, sigma))
            NN_b += (inten_scale(im) * saturation(im)) ** 2

    K_a = RPsum_a / (NN_a + 1)
    K_a = rgb2gray(K_a)
    K_a = zero_mean_total(K_a)
    K_a = wiener_dft(K_a, K_a.std(ddof=1)).astype(np.float32)
    K_b = RPsum_b / (NN_b + 1)
    K_b = rgb2gray(K_b)
    K_b = zero_mean_total(K_b)
    K_b = wiener_dft(K_b, K_b.std(ddof=1)).astype(np.float32)
    pce = prnu.pce(prnu.crosscorr_2d(K_a, K_b))['pce']
    if pce < 60:
        print("Warning: low PCE found:", pce)
    else:
        print("PCE:", pce)

    RPsum = RPsum_a + RPsum_b
    NN = NN_a + NN_b

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)
    return K, pce


def procedure(video_path: str, threads_count=os.cpu_count() - 2):
    if os.path.isdir(video_path.replace(".mp4", "/")):
        print("Skipping", video_path + ". Results already exist.")
        return

    mp4file = cv2.VideoCapture(video_path)
    fps = int(mp4file.get(cv2.CAP_PROP_FPS))
    tot_frames = int(mp4file.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_path + ": Fps:", str(fps) + ", frames count:", tot_frames)
    # seq = sequence_from_scenedetect(video_path)
    seq = [0, tot_frames]

    # fingerprint
    results = []
    for i in range(len(seq) - 1):
        print("Extracting..")
        max_frames = 500
        f = extract_frames(mp4file, list(range(seq[i], seq[i + 1]))[:max_frames])
        results.append(extract_and_test_multiple_aligned(f[:40], processes=threads_count)[1])
        results.append(extract_and_test_multiple_aligned(f[:80], processes=threads_count)[1])
        results.append(extract_and_test_multiple_aligned(f[:100], processes=threads_count)[1])
        results.append(extract_and_test_multiple_aligned(f[:200], processes=threads_count)[1])
        results.append(extract_and_test_multiple_aligned(f[:400], processes=threads_count)[1])
        results.append(extract_and_test_multiple_aligned(f, processes=threads_count)[1])

    return results


if __name__ == "__main__":
    for s in sys.argv[1::]:
        procedure(s)
