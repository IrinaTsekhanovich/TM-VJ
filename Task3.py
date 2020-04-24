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


def find_template(method, param, tmplts, image_path):
    img = cv2.imread(image_path, 0)
    img_res = np.array(method(img, param))
    min = 100000
    for template in tmplts:
        tmp = cv2.imread(template, 0)
        tmp_res = np.array(method(tmp, param))
        if np.linalg.norm(img_res - tmp_res) < min:
            min = np.linalg.norm(img_res - tmp_res)
            res = template
    return res


def hist(image, param):
    return cv2.calcHist(image, [0], None, [param], [0, 256])


def dct(image, p):
    return scipy.fft.dct(image, type=2, n=None, axis=-1, norm=None, overwrite_x=False)


def dft(image, p):
    return np.fft.fft(image)



def grad(image, ksize = 3):
    return cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize).mean(axis=0)

if __name__ == '__main__':

    count_of_templates = [7]
    params = [32]
    count_of_classes = 40
    stat_test = []
    for cnt in count_of_templates:
        templates = []
        images = []

        for d in range(1, count_of_classes+1):
            file_directory = f's{d}'
            for i in range(1, cnt + 1):
                templates.append(os.path.join(file_directory, f'{i}.pgm'))
            for i in range(cnt + 1, 11):
                images.append(os.path.join(file_directory, f'{i}.pgm'))
        # i = 0
        # for p in params:
        #     res = 0
        #     for image in images:
        #         tmp = find_template(grad, p, templates, image)
        #         if image[2] == tmp[2] and image[1] == tmp[1]:
        #             res += 1
        #     stat[i].append(round(res / ((10 - cnt) * count_of_classes) * 100))
        #     i += 1

        res = 0
        for image in images:
            tmp = find_template(hist, params[0], templates, image)
            if image[2] == tmp[2] and image[1] == tmp[1]:
                res += 1
            stat_test.append(res)
    for i in range(stat_test.__len__()):
        stat_test[i] = round((stat_test[i] / (i+1)) * 100)
    x = np.linspace(1, 3*40, 3*40)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_title('Histogram')
    ax.set_xlabel('Кол-во тестовых изображений')
    ax.set_ylabel('Верно классифицированные, %')
    ax.set_xlim(xmin=0, xmax=121)
    ax.set_ylim(ymin=0, ymax=110)
    ax.plot(x, stat_test, color='purple')
    plt.show()

    # # отрисовка графика зависимости от параметра и эталонов
    # fig = plt.figure(figsize=(9, 5))
    # ax = plt.axes(projection='3d')
    # c = ['blue', 'red', 'green', 'brown', 'purple', 'black']
    # i = 0
    # ax.set_title('Gradient')
    # ax.legend(loc='lower left')
    # ax.set_xlabel('Параметр')
    # ax.set_ylabel('Кол-во эталонов для одного класса')
    # ax.set_zlabel('Верно классифицированные, %')
    # # ax.set_xlim(xmin=0, xmax=11)
    # ax.set_ylim(ymin=0, ymax=10)
    # ax.set_zlim(zmin=0, zmax=110)
    # for t in params:
    #     ax.plot3D([t,t,t,t,t,t,t,t,t], count_of_templates, stat[i], color=c[i % 6])
    #     i += 1
    # # ax.plot3D(x, y, stat, 'co')
    # fig.tight_layout()
    # plt.show()


    # stat_res = [[],[],[],[],[]]
    # count_of_templates = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # params = [[2, 3, 4, 5]]
    # for w in count_of_templates:
    #     templates = []
    #     images = []
    #     for d in range(11, 16):
    #         file_directory = f's{d}'
    #         for i in range(1, w+1):
    #             templates.append(os.path.join(file_directory, f'{i}.pgm'))
    #         for i in range(w+1, 11):
    #             images.append(os.path.join(file_directory, f'{i}.pgm'))
    #
    #     for p in range(5):
    #         stat_res[p].append(0)
    #
    #     for image in images:
    #         tmp_sc = find_template(scale, templates, image)
    #         tmp_hist = find_template(hist, templates, image)
    #         tmp_dct = find_template(dct, templates, image)
    #         tmp_dft = find_template(dft, templates, image)
    #         tmp_grad = find_template(grad, templates, image)
    #
    #         if (image[2] == tmp_sc[2]):
    #             stat_res[0][w-1] += 1
    #         if (image[2] == tmp_hist[2]):
    #             stat_res[1][w-1] += 1
    #         if (image[2] == tmp_dct[2]):
    #             stat_res[2][w-1] += 1
    #         if (image[2] == tmp_dft[2]):
    #             stat_res[3][w-1] += 1
    #         if (image[2] == tmp_grad[2]):
    #             stat_res[4][w-1] += 1
    #     for p in range(5):
    #         stat_res[p][w-1] = round((stat_res[p][w-1]/((10 - w)*5))*100)

    # print(stat_res)

    #     fig = plt.figure()
    #     plt.subplot(332), plt.imshow(cv2.imread(image, 0), cmap='gray')
    #     plt.title('Test Image'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(323), plt.imshow(cv2.imread(tmp_sc, 0), cmap='gray')
    #     plt.title('Scale'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(324), plt.imshow(cv2.imread(tmp_hist, 0), cmap='gray')
    #     plt.title('Histogram'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(337), plt.imshow(cv2.imread(tmp_dct, 0), cmap='gray')
    #     plt.title('DCT'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(338), plt.imshow(cv2.imread(tmp_dft, 0), cmap='gray')
    #     plt.title('DFT'), plt.xticks([]), plt.yticks([])
    #     plt.subplot(339), plt.imshow(cv2.imread(tmp_grad, 0), cmap='gray')
    #     plt.title('Gradient'), plt.xticks([]), plt.yticks([])
    # plt.show()


    # l = ['Scale', 'Histogram', 'DCT', 'DFT', 'Gradient']
    # fig, ax = plt.subplots(figsize=(9, 5))
    # for t in range(5):
    #     ax.plot(count_of_templates, stat_res[t], label=l[t])
    # ax.set_title('Зависимость процента верных распознаваний от количества шаблонов')
    # ax.legend(loc='lower left')
    # ax.set_xlabel('Кол-во шаблонов в одном классе')
    # ax.set_ylabel('Верных распознавания, %')
    # ax.set_xlim(xmin=0, xmax=10)
    # ax.set_ylim(ymin=0, ymax=120)
    # fig.tight_layout()
    # plt.show()




