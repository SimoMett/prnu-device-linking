import numpy as np


def apply_lp_filter(img_array, cutoff):
    data_r = img_array[:, :, 0]
    data_r_fft = np.fft.fftshift(np.fft.fft2(data_r))
    data_r_fft[:1080, :(960 - cutoff)] = 1
    data_r_fft[:1080, (960 + cutoff):] = 1
    data_r_fft[:(540 - cutoff), :1920] = 1
    data_r_fft[(540 + cutoff):, :1920] = 1

    data_g = img_array[:, :, 1]
    data_g_fft = np.fft.fftshift(np.fft.fft2(data_g))
    data_g_fft[:1080, :(960 - cutoff)] = 1
    data_g_fft[:1080, (960 + cutoff):] = 1
    data_g_fft[:(540 - cutoff), :1920] = 1
    data_g_fft[(540 + cutoff):, :1920] = 1

    data_b = img_array[:, :, 2]
    data_b_fft = np.fft.fftshift(np.fft.fft2(data_b))
    data_b_fft[:1080, :(960 - cutoff)] = 1
    data_b_fft[:1080, (960 + cutoff):] = 1
    data_b_fft[:(540 - cutoff), :1920] = 1
    data_b_fft[(540 + cutoff):, :1920] = 1

    img_array[:, :, 0] = abs(np.fft.ifft2(data_r_fft))
    img_array[:, :, 1] = abs(np.fft.ifft2(data_g_fft))
    img_array[:, :, 2] = abs(np.fft.ifft2(data_b_fft))
    return img_array


def apply_lp_filter_grayscale(img_array, cutoff):
    height, width = img_array.shape

    data_fft = np.fft.fftshift(np.fft.fft2(img_array))
    data_fft[:height, :(int(width/2) - cutoff)] = 1
    data_fft[:height, (int(width/2) + cutoff):] = 1
    data_fft[:(int(height/2) - cutoff), :width] = 1
    data_fft[(int(height/2) + cutoff):, :width] = 1

    return abs(np.fft.ifft2(data_fft))
