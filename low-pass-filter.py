import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from PIL import Image
from scipy.signal import convolve2d
#from skimage.color import rgb2hsv, rgb2gray, rgb2yuv


def plot(data, title):
    plot.i += 1
    plt.subplot(2, 2, plot.i)
    plt.imshow(data)
    plt.gray()
    plt.title(title)

plot.i = 0


im = Image.open('out2.png')
data = np.array(im, dtype=int)
plot(data, 'Original')

data_r = data[:, :, 0]
data_r_fft = np.fft.fftshift(np.fft.fft2(data_r))
length = 8
data_r_fft[:1080, :(960-length)] = 1
data_r_fft[:1080, (960+length):] = 1
data_r_fft[:(540-length), :1920] = 1
data_r_fft[(540+length):, :1920] = 1

data_g = data[:, :, 1]
data_g_fft = np.fft.fftshift(np.fft.fft2(data_g))
data_g_fft[:1080, :(960-length)] = 1
data_g_fft[:1080, (960+length):] = 1
data_g_fft[:(540-length), :1920] = 1
data_g_fft[(540+length):, :1920] = 1

data_b = data[:, :, 2]
data_b_fft = np.fft.fftshift(np.fft.fft2(data_b))
data_b_fft[:1080, :(960-length)] = 1
data_b_fft[:1080, (960+length):] = 1
data_b_fft[:(540-length), :1920] = 1
data_b_fft[(540+length):, :1920] = 1

data[:, :, 0] = abs(np.fft.ifft2(data_r_fft))
data[:, :, 1] = abs(np.fft.ifft2(data_g_fft))
data[:, :, 2] = abs(np.fft.ifft2(data_b_fft))

plot(data, 'LP transformed')

plt.show()