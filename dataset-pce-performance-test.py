import re
import sys
import os
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from tqdm import tqdm
import params
import prnu
import scene_detect
from extract_frames import extract_frames
from prnu import inten_sat_compact, noise_extract_compact, inten_scale, saturation, rgb2gray, zero_mean_total, \
    wiener_dft
from scene_detect import sequence_from_scenedetect


def save_results(output_path, pce_rot, stats_pce):
    os.makedirs(output_path, exist_ok=True)

    with open(output_path + "pce.csv", "w") as output_file:
        for row in pce_rot:
            output_file.write(",".join(("{:.1f}".format(i) for i in row)) + "\n")

    with open(output_path + "stats_pce.csv", "w") as output_file:
        output_file.write("TPR:," + ",".join((str(i) for i in stats_pce['tpr'])) + "\n")
        output_file.write("FPR:," + ",".join((str(i) for i in stats_pce['fpr'])) + "\n")
        output_file.write("TH:," + ",".join((str(i) for i in stats_pce['th'])) + "\n")
        output_file.write("AUC:," + str(stats_pce['auc']) + "\n")
        output_file.write("EER:," + str(stats_pce['eer']) + "\n")

    # save_as_pickle(output_path + "full_results.pickle", (aligned_cc, stats_cc, pce_rot, stats_pce))


def pce_rot_func(_fp_k, _res_w):
    return prnu.pce(prnu.crosscorr_2d(_fp_k, _res_w))['pce']


def extract_and_test_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                                      batch_size=cpu_count(), tqdm_str: str = '') -> np.ndarray:
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
    return K


def procedure(video_path: str, frames_count):
    threads_count = cpu_count() - 1 if cpu_count() != 1 else 1
    #threads_count = 1
    mp4file = cv2.VideoCapture(video_path)
    fps = int(mp4file.get(cv2.CAP_PROP_FPS))
    tot_frames = int(mp4file.get(cv2.CAP_PROP_FRAME_COUNT))
    # print(video_path + ": Fps:", str(fps) + ", frames count:", tot_frames)
    seq = [0, tot_frames]

    # fingerprint
    clips_fingerprints_k = []
    for i in range(len(seq) - 1):
        end = seq[i] + frames_count if frames_count is not None else seq[i + 1]
        f = extract_frames(mp4file, list(range(seq[i], end)))
        print("Computing fingerprint from", end - seq[i], "frames..")

        clips_fingerprints_k.append(extract_and_test_multiple_aligned(f, processes=threads_count))
    return


def compute_pce(clips_fingerprints_k, residuals_w):
    pce_rot = np.zeros((len(clips_fingerprints_k), len(residuals_w)))
    pp = [[None for __ in clips_fingerprints_k] for _ in residuals_w]
    pool = Pool(os.cpu_count() - 1 if os.cpu_count() != 1 else 1)
    for i, fp_k in enumerate(clips_fingerprints_k):
        for j, res_w in enumerate(residuals_w):
            pp[i][j] = pool.apply_async(pce_rot_func, (fp_k, res_w,))
    for i, fp_k in enumerate(clips_fingerprints_k):
        for j, res_w in enumerate(residuals_w):
            pce_rot[i, j] = pp[i][j].get()
    pool.close()
    return pce_rot


if __name__ == "__main__":
    s = sys.argv[1]
    procedure(s, 20)
    procedure(s, 40)
    procedure(s, 60)
    procedure(s, 80)
    procedure(s, 120)
    procedure(s, 240)
    procedure(s, None)
