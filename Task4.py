import cv2
import os
import numpy as np
import scipy
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d


def avrsum(img, i, j, step):
    sum = 0
    for k in range(i, i + step):
        for l in range(j, j + step):
            sum += img[k, l]
    return sum / (step * step)


def scale(img, step=2):
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


def hist(image):
    return cv2.calcHist(image, [0], None, [16], [0, 256])


def dct(image):
    return scipy.fft.dct(image, type=2, n=3, axis=-1, norm=None, overwrite_x=False)


def dft(image):
    return np.fft.fft(image,3)


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


def grad(image, ksize=5):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize).mean(axis=0)

if __name__ == '__main__':
    stat_test = []
    cnt = 40
    count_of_templates = [6]
    for w in count_of_templates:
        templates = []
        images = []
        for d in range(1, cnt+1):
            file_directory = f's{d}'
            for i in range(1, w+1):
                templates.append(os.path.join(file_directory, f'{i}.pgm'))
            for i in range(w+1, 11):
                images.append(os.path.join(file_directory, f'{i}.pgm'))
        # stat_res.append(0)
        res = 0
        for image in images:
            tmp_sc = find_template(scale, templates, image)
            tmp_hist = find_template(hist, templates, image)
            tmp_dct = find_template(dct, templates, image)
            tmp_dft = find_template(dft, templates, image)
            tmp_grad = find_template(grad, templates, image)
            tmp_parall = parall([tmp_sc, tmp_hist, tmp_dct, tmp_dft, tmp_grad])
            if image[2] == tmp_parall[2] and image[1] == tmp_parall[1]:
                res += 1
            stat_test.append(res)

            # if (image[1] == tmp_sc[1]):
            #     stat_res[0][w-1] += 1
            # if (image[1] == tmp_hist[1]):
            #     stat_res[1][w-1] += 1
            # if (image[1] == tmp_dct[1]):
            #     stat_res[2][w-1] += 1
            # if (image[1] == tmp_dft[1]):
            #     stat_res[3][w-1] += 1
            # if (image[1] == tmp_grad[1]):
            #     stat_res[4][w-1] += 1


        #     if image[1] == tmp_parall[1] and image[2] == tmp_parall[2]:
        #         stat_res[w - 1] += 1
        # stat_res[w-1] = round((stat_res[w-1]/((10 - w)*cnt))*100)

    for i in range(stat_test.__len__()):
        stat_test[i] = round((stat_test[i] / (i + 1)) * 100)
    x = np.linspace(1, 4 * 40, 4 * 40)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('Parallel')
    ax.set_xlabel('Кол-во тестовых изображений')
    ax.set_ylabel('Верно классифицированные, %')
    ax.set_xlim(xmin=0, xmax=161)
    ax.set_ylim(ymin=0, ymax=110)
    ax.plot(x, stat_test, color='purple')
    plt.show()

    #
    #     fig = plt.figure()
    #     plt.subplot(332), plt.imshow(cv2.imread(image, 0), cmap='gray')
    #     plt.title('Test Image'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(334), plt.imshow(cv2.imread(tmp_sc, 0), cmap='gray')
    #     plt.title('Scale'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(335), plt.imshow(cv2.imread(tmp_parall, 0), cmap='gray')
    #     plt.title('Parallel'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(336), plt.imshow(cv2.imread(tmp_hist, 0), cmap='gray')
    #     plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(337), plt.imshow(cv2.imread(tmp_dct, 0), cmap='gray')
    #     plt.title('DCT'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(338), plt.imshow(cv2.imread(tmp_dft, 0), cmap='gray')
    #     plt.title('DFT'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(339), plt.imshow(cv2.imread(tmp_grad, 0), cmap='gray')
    #     plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    # plt.show()

    # l = ['Scale', 'Histogram', 'DCT', 'DFT', 'Gradient', 'Parallel']
    # c = ['blue', 'red', 'green', 'brown', 'purple', 'black']
    # fig = plt.figure(figsize=(9, 5))
    # ax = plt.axes(projection='3d')
    # for t in range(6):
    #     ax.plot3D(count_of_templates, [t+1, t+1, t+1, t+1, t+1, t+1, t+1, t+1], stat_res[t], label=l[t], color=c[t])
    # ax.set_title('Зависимость верных распознаваний от количества шаблонов')
    # ax.legend(loc='upper left')
    # ax.set_xlabel('Кол-во шаблонов в одном классе')
    # ax.set_zlabel('Верных распознавания, %')
    # ax.set_xlim(left=0, right=10)
    # ax.set_ylim(bottom=0, top=7)
    # ax.set_zlim(bottom=0, top=120)
    # fig.tight_layout()
    # plt.show()


