import cv2
from PIL import Image
import numpy as np
import os
PATH = "../videos/accessory/"
class DHash(object):
    @staticmethod
    def calculate_hash(image):
        """
        计算图片的dHash值
        :param image: PIL.Image
        :return: dHash值,string类型
        """
        difference = DHash.__difference(image)
        # 转化为16进制(每个差值为一个bit,每8bit转为一个16进制)
        decimal_value = 0
        hash_string = ""
        for index, value in enumerate(difference):
            if value:  # value为0, 不用计算, 程序优化
                decimal_value += value * (2 ** (index % 8))
            if index % 8 == 7:  # 每8位的结束
                hash_string += str(hex(decimal_value)[2:].rjust(2, "0"))  # 不足2位以0填充。0xf=>0x0f
                decimal_value = 0
        return hash_string

    @staticmethod
    def hamming_distance(first, second):
        """
        计算两张图片的汉明距离(基于dHash算法)
        :param first: Image或者dHash值(str)
        :param second: Image或者dHash值(str)
        :return: hamming distance. 值越大,说明两张图片差别越大,反之,则说明越相似
        """
        # A. dHash值计算汉明距离
        if isinstance(first, str):
            return DHash.__hamming_distance_with_hash(first, second)

        # B. image计算汉明距离
        hamming_distance = 0
        image1_difference = DHash.__difference(first)
        image2_difference = DHash.__difference(second)
        for index, img1_pix in enumerate(image1_difference):
            img2_pix = image2_difference[index]
            if img1_pix != img2_pix:
                hamming_distance += 1
        return hamming_distance

    @staticmethod
    def __difference(image):
        """
        *Private method*
        计算image的像素差值
        :param image: PIL.Image
        :return: 差值数组。0、1组成
        """
        resize_width = 9
        resize_height = 8
        # 1. resize to (9,8)
        smaller_image = image.resize((resize_width, resize_height))
        # 2. 灰度化 Grayscale
        grayscale_image = smaller_image.convert("L")
        # 3. 比较相邻像素
        pixels = list(grayscale_image.getdata())
        difference = []
        for row in range(resize_height):
            row_start_index = row * resize_width
            for col in range(resize_width - 1):
                left_pixel_index = row_start_index + col
                difference.append(pixels[left_pixel_index] > pixels[left_pixel_index + 1])
        return difference


    @staticmethod
    def __hamming_distance_with_hash(dhash1, dhash2):
        """
        *Private method*
        根据dHash值计算hamming distance
        :param dhash1: str
        :param dhash2: str
        :return: 汉明距离(int)
        """
        difference = (int(dhash1, 16)) ^ (int(dhash2, 16))
        return bin(difference).count("1")
def a(cap):
    # cap = cv2.VideoCapture('../videos/cloth/' + filename)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    size = (int(width), int(height))
    fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')

    # if frame_count/fps <15:
    #     print(filename)
    #     continue

    ret, cframe = cap.read()
    cImage = Image.fromarray(cv2.cvtColor(cframe, cv2.COLOR_BGR2RGB))
    dis = []
    i = 0
    while cap.isOpened():
        # cap.set(cv2.CAP_PROP_POS_FRAMES, i+8)
        ret, fframe = cap.read()
        if ret == True:
            fImage = Image.fromarray(cv2.cvtColor(fframe, cv2.COLOR_BGR2RGB))
            d = DHash.hamming_distance(cImage, fImage)
            dis.append(d)
            # print(d)
            # ht = np.hstack((cframe,fframe))
            # cv2.imshow("aa",ht)
            # if cv2.waitKey(0) & 0xFF == ord('q'):
            #     print()
            cImage = fImage
            cframe = fframe
            # i+=8
        else:
            break
    mean = sum(dis) / len(dis)
    # print(mean)

    cutpoint = []

    if mean < 3:
        th = 10
        for i in range(2, len(dis) - 2):
            if dis[i] > th and dis[i - 2] * 2 < dis[i] and dis[i - 1] * 2 < dis[i] and dis[i + 1] * 2 < dis[i] and dis[
                i + 2] * 2 < dis[i]:
                cutpoint.append(i)
    elif mean < 8:
        th = 18
        for i in range(2, len(dis) - 2):
            if dis[i] > th and (dis[i] - dis[i - 2]) > 10 and (dis[i] - dis[i - 1]) > 10 and (
                    dis[i] - dis[i + 1]) > 10 and (dis[i] - dis[i + 2]) > 10:
                cutpoint.append(i)
    else:
        th = 22
        for i in range(2, len(dis) - 2):
            if dis[i] > th and ((dis[i] - dis[i - 2]) > 13 and (dis[i] - dis[i - 1]) > 13) or (
                    (dis[i] - dis[i + 1]) > 13 and (dis[i] - dis[i + 2]) > 13):
                cutpoint.append(i)

    if len(cutpoint) == 0:
        cutpoint.append(int(frame_count / 3))
        cutpoint.append(int(2 * frame_count / 3))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cutpoint.append(frame_count)
    # print(cutpoint)
    m=0
    while m < len(cutpoint) - 1:
        if abs(cutpoint[m] - cutpoint[m + 1]) <= 10:
            cutpoint.remove(cutpoint[m])
            m -= 1
        m += 1
    cutpoint.insert(0, 0)
    cutShot = []
    for i, value in enumerate(cutpoint):
        if i == len(cutpoint) - 1:
            break
        cutShot.append((value, cutpoint[i + 1] - 1))
    # print(cutShot)
    return cutShot



#
# cap2=cv2.VideoCapture(PATH+"3.mp4")
# length = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
# print(length)
# a(cap2)

