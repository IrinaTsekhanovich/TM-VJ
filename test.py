import scipy
import random
import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

def grad(image):
    return np.gradient(image)

def rndm_points(p):
    res = []
    for i in range(p):
        res.append([random.randint(20, 90), random.randint(20, 70)])
    return res


def rndm_lines(p, d):
    res = []
    for i in range(p):
        x = random.randint(10, 100)
        for j in range(d):
            res.append([x, random.randint(10, 80)])
    return res


def rndm_square(s1, s2):
    res = []
    for i in range(1,111-s1,s1):
        for j in range(1,91-s2,s2):
            res.append([i, j])
    return res

rndm_arr = rndm_points(1500)

rndm1_arr = rndm_lines(40, 30)

rndm2_arr = rndm_square(4,6)

def rndm(image, arr = rndm2_arr):
    res = []
    for i in range(arr.__len__()):
        res.append(image[arr[i][0], arr[i][1]])
    return res


def hist(image):
    return cv2.calcHist(image, [0], None, [16], [0, 256])


def dft(image):
    return np.fft.fft(image)


def grad(image, ksize = 5):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize).mean(axis=0)


def avrsum(img, i, j, step):
    sum = 0
    for k in range(i, i + step):
        for l in range(j, j + step):
            sum += img[k, l]
    return sum / (step * step)

def parall(arr):
    tmpx = []
    for i in arr:
        tmpx.append(i[1])
    fun = {}
    for x in tmpx:
        if fun.get(x, 0) == 0:
            fun.update({x: 1})
        else:
            fun.update({x: fun.get(x)+1})
    max = 1
    for key, value in fun.items():
        if value > max:
            max = value
            res = key
    for i in arr:
        if i[1] == res:
            return i

def scale(img, step=4):
    w, h = img.shape[::-1]
    tmp = []
    n = int((w / step) * (h / step))
    for i in range(0, h - step + 1, step):
        for j in range(0, w - step + 1, step):
            tmp.append(avrsum(img, i, j, step))
    return tmp


def find_template(method, tmplts, image_path):
    img = cv2.imread(image_path, 0)
    img_res = np.array(method(img,))
    min = 100000
    for template in tmplts:
        tmp = cv2.imread(template, 0)
        tmp_res = np.array(method(tmp))
        if np.linalg.norm(img_res - tmp_res) < min:
            min = np.linalg.norm(img_res - tmp_res)
            res = template
    return res


def dct(image):
    return scipy.fft.dct(image, type=2, n=None, axis=-1, norm=None, overwrite_x=False)

if __name__ == '__main__':
    cnt = 10
    templates = []
    images = []
    for d in range(1, cnt+1):
        file_directory = f's{d}'
        for i in range(1, 7):
            templates.append(os.path.join(file_directory, f'{i}.pgm'))
        for i in range(7, 8):
            images.append(os.path.join(file_directory, f'{i}.pgm'))
    for image in images:
        img = cv2.imread(image, 0)
        df_ph = np.asarray(20 * np.log(np.abs(dft(img))))[0:5, 0:5]
        gr_ph = grad(img)
        sc_ph = np.asarray(scale(img)).reshape(28, 23)
        h_ph = hist(img).reshape(16, )
        dc_ph = np.asarray(20 * np.log(np.abs(dct(img))))[0:6, 0:6]

        tmp_sc = find_template(scale, templates, image)
        tmp_hist = find_template(hist, templates, image)
        tmp_dct = find_template(dct, templates, image)
        tmp_dft = find_template(dft, templates, image)
        tmp_grad = find_template(grad, templates, image)
        tmp_parall = parall([tmp_sc, tmp_hist, tmp_dct, tmp_dft, tmp_grad])


        fig = plt.figure(figsize=(7, 5))
        x = np.linspace(0, 255, 16)
        y = np.arange(1, 93)
        plt.subplot(336), plt.bar(x, h_ph, width=4)
        plt.title('Hist')
        plt.subplot(338), plt.imshow(dc_ph, cmap='gray')
        plt.title('DCT'), plt.xticks([]), plt.yticks([])
        plt.subplot(339), plt.plot(y, gr_ph)
        # plt.title('Grad')
        plt.subplot(332), plt.imshow(img, cmap='gray')
        plt.title('Test Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(334), plt.imshow(sc_ph, cmap='gray')
        plt.title('Scale'), plt.xticks([]), plt.yticks([])
        plt.subplot(337), plt.imshow(df_ph, cmap='gray')
        plt.title('DFT'), plt.xticks([]), plt.yticks([])

        fig = plt.figure(figsize=(7, 5))
        plt.subplot(332), plt.imshow(cv2.imread(image, 0), cmap='gray')
        plt.title('Test Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(334), plt.imshow(cv2.imread(tmp_sc, 0), cmap='gray')
        plt.title('Scale'), plt.xticks([]), plt.yticks([])
        plt.subplot(335), plt.imshow(cv2.imread(tmp_parall, 0), cmap='gray')
        plt.title('Parallel'), plt.xticks([]), plt.yticks([])
        plt.subplot(336), plt.imshow(cv2.imread(tmp_hist, 0), cmap='gray')
        plt.title('Histogram'), plt.xticks([]), plt.yticks([])
        plt.subplot(337), plt.imshow(cv2.imread(tmp_dct, 0), cmap='gray')
        plt.title('DCT'), plt.xticks([]), plt.yticks([])
        plt.subplot(338), plt.imshow(cv2.imread(tmp_dft, 0), cmap='gray')
        plt.title('DFT'), plt.xticks([]), plt.yticks([])
        plt.subplot(339), plt.imshow(cv2.imread(tmp_grad, 0), cmap='gray')
        plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    plt.show()


