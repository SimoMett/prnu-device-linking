import re
import sys
import os
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
import pickle as pk
from tqdm import tqdm
import params
import prnu
import scene_detect
from extract_frames import extract_frames
from prnu import inten_sat_compact, noise_extract_compact, inten_scale, saturation, rgb2gray, zero_mean_total, \
    wiener_dft, extract_multiple_aligned
from scene_detect import sequence_from_scenedetect


def save_as_pickle(filename: str, object):
    with open(filename, "wb") as output_file:
        pk.dump(object, output_file)
    print("Generated", filename)
    return


def save_results(video_path, aligned_cc, stats_cc, pce_rot, stats_pce):
    output_path = video_path.replace(".mp4", "/")
    os.makedirs(output_path, exist_ok=True)

    if aligned_cc is not None:
        with open(output_path + "aligned_cc.csv", "w") as output_file:
            for row in aligned_cc:
                output_file.write(",".join(("{:.1f}".format(i) for i in row)) + "\n")

    with open(output_path + "pce.csv", "w") as output_file:
        for row in pce_rot:
            output_file.write(",".join(("{:.1f}".format(i) for i in row)) + "\n")

    if stats_cc is not None:
        with open(output_path + "stats_cc.csv", "w") as output_file:
            output_file.write("TPR:," + ",".join((str(i) for i in stats_cc['tpr'])) + "\n")
            output_file.write("FPR:," + ",".join((str(i) for i in stats_cc['fpr'])) + "\n")
            output_file.write("TH:," + ",".join((str(i) for i in stats_cc['th'])) + "\n")
            output_file.write("AUC:," + str(stats_cc['auc']) + "\n")
            output_file.write("EER:," + str(stats_cc['eer']) + "\n")

    with open(output_path + "stats_pce.csv", "w") as output_file:
        output_file.write("TPR:," + ",".join((str(i) for i in stats_pce['tpr'])) + "\n")
        output_file.write("FPR:," + ",".join((str(i) for i in stats_pce['fpr'])) + "\n")
        output_file.write("TH:," + ",".join((str(i) for i in stats_pce['th'])) + "\n")
        output_file.write("AUC:," + str(stats_pce['auc']) + "\n")
        output_file.write("EER:," + str(stats_pce['eer']) + "\n")

    save_as_pickle(output_path + "full_results.pickle", (aligned_cc, stats_cc, pce_rot, stats_pce))


def pce_rot_func(_fp_k, _res_w):
    return prnu.pce(prnu.crosscorr_2d(_fp_k, _res_w))['pce']


def multiprocess_get_sum_and_nn(imgs, args_list, processes, batch_size, tqdm_str):
    h, w, ch = imgs[0].shape
    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)
    pool = Pool(processes=processes)

    for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                           desc=(tqdm_str + ' (1/2)'), dynamic_ncols=True):
        nni = pool.map(inten_sat_compact, args_list[batch_idx0:batch_idx0 + batch_size])
        # Do not use np.sum over axis=0. It's slower
        for ni in nni:
            NN += ni
        del nni

    for batch_idx0 in tqdm(np.arange(start=0, step=batch_size, stop=len(imgs)), disable=tqdm_str == '',
                           desc=(tqdm_str + ' (2/2)'), dynamic_ncols=True):
        wi_list = pool.map(noise_extract_compact, args_list[batch_idx0:batch_idx0 + batch_size])
        for wi in wi_list:
            RPsum += wi
        del wi_list
    pool.close()
    return RPsum, NN


def get_sum_and_nn(imgs, levels, sigma, tqdm_str):
    h, w, ch = imgs[0].shape
    RPsum = np.zeros((h, w, ch), np.float32)
    NN = np.zeros((h, w, ch), np.float32)
    for im in tqdm(imgs, disable=tqdm_str is None, desc="1/2" + tqdm_str, dynamic_ncols=True):
        RPsum += noise_extract_compact((im, levels, sigma))
        NN += (inten_scale(im) * saturation(im)) ** 2
    return RPsum, NN


def extract_and_test_multiple_aligned(imgs: list, levels: int = 4, sigma: float = 5, processes: int = None,
                                      batch_size=cpu_count(), tqdm_str: str = '') -> (np.ndarray, float):
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
        block_a = imgs[:int(len(imgs) / 2)]
        args_list = [(im, levels, sigma) for im in block_a]
        RPsum_a, NN_a = multiprocess_get_sum_and_nn(block_a, args_list, processes, batch_size, tqdm_str)

        # Second half of imgs
        block_b = imgs[int(len(imgs) / 2):]
        args_list = [(im, levels, sigma) for im in block_b]
        RPsum_b, NN_b = multiprocess_get_sum_and_nn(block_b, args_list, processes, batch_size, tqdm_str)

    else:  # Single process
        # First half
        RPsum_a, NN_a = get_sum_and_nn(imgs[:int(len(imgs) / 2)], levels, sigma, tqdm_str)

        # Second half
        RPsum_b, NN_b = get_sum_and_nn(imgs[int(len(imgs) / 2):], levels, sigma, tqdm_str)

    K_a = rgb2gray(RPsum_a / (NN_a + 1))
    K_a = zero_mean_total(K_a)
    K_a = wiener_dft(K_a, K_a.std(ddof=1)).astype(np.float32)

    K_b = rgb2gray(RPsum_b / (NN_b + 1))
    K_b = zero_mean_total(K_b)
    K_b = wiener_dft(K_b, K_b.std(ddof=1)).astype(np.float32)

    pce = prnu.pce(prnu.crosscorr_2d(K_a, K_b))['pce']

    RPsum = RPsum_a + RPsum_b
    NN = NN_a + NN_b

    K = RPsum / (NN + 1)
    K = rgb2gray(K)
    K = zero_mean_total(K)
    K = wiener_dft(K, K.std(ddof=1)).astype(np.float32)
    return K, pce


def procedure(video_path: str, threads_count, frames_count=4):
    if os.path.isdir(video_path.replace(".mp4", "/")):
        print("Skipping", video_path + ". Results already exist.")
        return

    mp4file = cv2.VideoCapture(video_path)
    fps = int(mp4file.get(cv2.CAP_PROP_FPS))
    tot_frames = int(mp4file.get(cv2.CAP_PROP_FRAME_COUNT))
    print(video_path + ": Fps:", str(fps) + ", frames count:", tot_frames)
    seq = sequence_from_scenedetect(video_path)

    # ground truth
    seq_idx = int(re.search(r'\d+', video_path.split("/")[-1]).group()) - 1
    clips_seq = params.sequences[seq_idx]
    ground_truth = prnu.gt(clips_seq, clips_seq)

    # print("Extracting residuals from chosen samples")
    # samples = extract_frames(mp4file, seq[:-1])
    # pool = Pool(os.cpu_count() - 1 if os.cpu_count() != 1 else 1)
    # residuals_w = pool.map(prnu.extract_single, samples)
    # pool.close()

    # fingerprint
    clips_fingerprints_k = []

    for i in range(len(seq) - 1):
        print("Extracting frames from clip", i + 1)
        # end = seq[i + 1]
        if frames_count == 'end':
            end = seq[i+1]-1
        else:
            end = seq[i] + frames_count
        assert end < seq[i+1]
        f = extract_frames(mp4file, list(range(seq[i], end)))
        print("Computing fingerprint..")
        K = extract_multiple_aligned(f, processes=threads_count, batch_size=threads_count)
        clips_fingerprints_k.append(K)

    # print("Cross-correlation")
    # aligned_cc = prnu.aligned_cc(np.array(clips_fingerprints_k), np.array(residuals_w))['cc']
    # stats_cc = prnu.stats(aligned_cc, ground_truth)

    print("Peak to correlation energy")
    pce_rot = compute_pce(clips_fingerprints_k, clips_fingerprints_k)
    stats_pce = prnu.stats(pce_rot, ground_truth)
    save_results(video_path, None, None, pce_rot, stats_pce)
    return


def compute_pce(clips_fingerprints_k, residuals_w):
    pce_rot = np.zeros((len(clips_fingerprints_k), len(residuals_w)))
    pp = [[None for _ in range(len(clips_fingerprints_k))] for _ in range(len(residuals_w))]
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
    for s in sys.argv[1::]:
        procedure(s, cpu_count() - 2, 200)
