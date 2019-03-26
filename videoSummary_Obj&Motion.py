import tensorflow as tf
import numpy as np
import os
from scipy import misc
import argparse
# import imageio
# import pylab
import cv2
import imutils
import sys
import json

from typing import Tuple

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
g_mean = np.array(([126.88, 120.24, 112.19])).reshape([1, 1, 3])

fgbg=cv2.createBackgroundSubtractorMOG2()
kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
stasaliency = cv2.saliency.StaticSaliencyFineGrained_create()

# RGBA转RGB,用Alpha通道的数值去乘RGB三个通道的数值
def rgba2rgb(img):
    return img[:, :, :3] * np.expand_dims(img[:, :, 3], 2)

def find_index_list(index_heap, length, num_peak):
    span = length//num_peak + 1
    hash_index = [0 for i in range(num_peak)]
    final_l = []
    item = index_heap.pop()
    cover_index = item[1]

    while final_l.__len__() < num_peak:
        print(item)
        index = item[1]
        if hash_index[index // span] == 0 and index > 75 // num_peak and index < length - 75 // num_peak:
            flag = True
            for peek in final_l:
                if abs(index - peek) < 30:
                    flag = False
            if flag:
                final_l.append(index)
                hash_index[index // span] = 1
        item = index_heap.pop()

    final_l.sort()
    print('=====================Final_Peak_list====================')
    print(final_l)
    return final_l, cover_index

def staticSaliency(frame):

    (success, saliencyMap) = stasaliency.computeSaliency(frame)
    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # cv2.imshow("staticMap", saliencyMap)
    # cv2.imshow("Thresh", threshMap)
    return np.sum(threshMap) / (threshMap.shape[0] * threshMap.shape[1] * 255)

def foreground(frame):
    fgmask=fgbg.apply(frame)
    fgmask=cv2.morphologyEx(fgmask,cv2.MORPH_OPEN,kernel)
    fgmask = cv2.threshold(fgmask.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # cv2.imshow("fgmask",fgmask)
    return np.sum(fgmask) / (fgmask.shape[0] * fgmask.shape[1] * 255)

def main(video_name):

    output_folder = "./test_output_Obj&Motion/"
    video_path = "./video/"
    video = video_path + video_name + '.mp4'
    summary = output_folder + 'Sum_' + video_name + '.avi'

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    #分配GPU用量
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)


    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./salience_model'))
        image_batch = tf.get_collection('image_batch')[0]
        pred_mattes = tf.get_collection('mask')[0]

        cap = cv2.VideoCapture(video)
        shape = (int(cap.get(3)),int(cap.get(4)))
        fps = cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, image = cap.read()
        print(shape)
        index_heap = MaxHeap()

        print('=====================Start Analysis=====================\n' + 'The Length of vedio is: ' + str(length))

        #file = open(output_folder + video_name + '.json', 'w')
        if cap.isOpened():
            for m in range(0, length):
                cap.set(cv2.CAP_PROP_POS_FRAMES, m)
                ret, image = cap.read()
                image = imutils.resize(image, width=500)
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                rgb = image
                if rgb.shape[2] == 4:
                    rgb = rgba2rgb(rgb)
                origin_shape = rgb.shape[:2]
                rgb = np.expand_dims(
                    misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)

                feed_dict = {image_batch: rgb}
                pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
                final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
                #显著度最高为255,最低为0,归一化
                objSal = np.sum(final_alpha/255) / (final_alpha.shape[0] * final_alpha.shape[1])


                # staSal = staticSaliency(image)
                # moSal = foreground(image)

                index_heap.add((objSal, m))

                # result_dict = {'objSal': result}
                # jsObj = json.dumps(result_dict)
                # file.write(jsObj)
                # file.write('\n')
                # print(m, result)

            # Find the largest n peaks

            num_peak = 5
            final_frame = find_index_list(index_heap, length, num_peak)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            videowriter = cv2.VideoWriter(summary, fourcc, fps, shape)
            print('=====================Start making Summary====================')
            for frame in final_frame:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame-75//num_peak)
                for i in range(150//num_peak):
                    ret, image = cap.read()
                    # cv2.imshow('Windows', image)
                    # cv2.waitKey(0)
                    videowriter.write(image)


        cap.release()
        videowriter.release()
        cv2.destroyAllWindows()
        print('=====================End====================')





#大顶堆

class MaxHeap(object):

    def __init__(self):
        self._data = []
        self._count = len(self._data)

    def size(self):
        return self._count

    def isEmpty(self):
        return self._count == 0

    def add(self, item):
        # 插入元素入堆
        self._data.append(item)
        self._count += 1
        self._shiftup(self._count - 1)

    def pop(self):
        # 出堆
        if self._count > 0:
            ret = self._data[0]
            self._data[0] = self._data[self._count - 1]
            self._count -= 1
            self._shiftDown(0)
            return ret

    def _shiftup(self, index):
        # 上移self._data[index]，以使它不大于父节点
        parent = (index - 1) >> 1
        while index > 0 and self._data[parent] < self._data[index]:
            # swap
            self._data[parent], self._data[index] = self._data[index], self._data[parent]
            index = parent
            parent = (index - 1) >> 1

    def _shiftDown(self, index):
        # 上移self._data[index]，以使它不小于子节点
        j = (index << 1) + 1
        while j < self._count:
            # 有子节点
            if j + 1 < self._count and self._data[j + 1] > self._data[j]:
                # 有右子节点，并且右子节点较大
                j += 1
            if self._data[index] >= self._data[j]:
                # 堆的索引位置已经大于两个子节点，不需要交换了
                break
            self._data[index], self._data[j] = self._data[j], self._data[index]
            index = j
            j = (index << 1) + 1

    # def largeset_n(self, n):
    #     if n>self._count:
    #         n=self._count
    #     res = []
    #     for i in n:
    #         res.append(self._data[0])
    #         self.pop()
    #     return res




def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--video_name', type=str,
                        help='input video name', default=None)
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    video_name = args.video_name
    main(video_name)

