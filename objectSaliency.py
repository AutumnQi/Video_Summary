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

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
g_mean = np.array(([126.88,120.24,112.19])).reshape([1,1,3])
output_folder = "./test_output/"

#RGBA转RGB,用Alpha通道的数值去乘RGB三个通道的数值
def rgba2rgb(img):
	return img[:,:,:3]*np.expand_dims(img[:,:,3],2)

def detect(video):
	
	if not os.path.exists(output_folder):
		os.mkdir(output_folder)	
	
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
	with tf.Session(config=tf.ConfigProto(gpu_options = gpu_options)) as sess:
		saver = tf.train.import_meta_graph('./meta_graph/my-model.meta')
		saver.restore(sess,tf.train.latest_checkpoint('./salience_model'))
		image_batch = tf.get_collection('image_batch')[0]
		pred_mattes = tf.get_collection('mask')[0]

		cap = cv2.VideoCapture(video)
		length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
		print(length)
		file = open(output_folder + video_name + '.json', 'w')
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
				# print(origin_shape)
				rgb = np.expand_dims(
					misc.imresize(rgb.astype(np.uint8), [320, 320, 3], interp="nearest").astype(np.float32) - g_mean, 0)

				feed_dict = {image_batch: rgb}
				pred_alpha = sess.run(pred_mattes, feed_dict=feed_dict)
				final_alpha = misc.imresize(np.squeeze(pred_alpha), origin_shape)
				result = np.sum(final_alpha) / (final_alpha.shape[0] * final_alpha.shape[1] * 255)

				result_dict = {'objSal': result}
				jsObj=json.dumps(result_dict)
				file.write(jsObj)
				file.write('\n')
				print(m, result)

		cap.release()

def parse_arguments(argv):
	parser = argparse.ArgumentParser()

	parser.add_argument('--video_name', type=str,
		help='input video name',default = None)
	return parser.parse_args(argv)


# if __name__ == '__main__':
video_path = "./video/"
args=parse_arguments(sys.argv[1:])
video_name=args.video_name

detect(video_path+video_name+".mp4")

cv2.waitKey(0)
cv2.destroyAllWindows()