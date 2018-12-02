


from random import shuffle
import os
import sys
import cv2
import numpy as np
import tensorflow as tf

def load_image(filePath):
	img = cv2.imread(filePath)
	if img is None:
		return None
	img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def createDataRecord(out_filename, filePaths, labels):
	writer = tf.python_io.TFRecordWriter(out_filename)
	for x in range(len(filePaths)):
		img = load_image(filePaths[x])
		label = labels[x]
		if img is None:
			continue

		feature = {
			'image_raw': _bytes_feature(img.tostring()),
			'label': _int64_feature(label)
		}

		example = tf.train.Example(features=tf.train.Features(feature=feature))
		writer.write(example.SerializeToString())	
	writer.close()
	sys.stdout.flush()

def split_trainTest(imgsAll, labelAll, splitRatio = 0.20):
	dataZip = list(zip(imgsAll, labelAll))
	shuffle(dataZip)
	imgsAll, labelAll = zip(*dataZip)
	splitPoint = int(len(imgsAll)*splitRatio)

	trainImgs = imgsAll[:int(len(imgsAll) - splitPoint)]
	trainLabel = labelAll[:int(len(imgsAll) - splitPoint)]
	testImgs = imgsAll[int(len(imgsAll)-splitPoint):]
	testLabel = labelAll[int(len(imgsAll)-splitPoint):]

	return trainImgs, testImgs, trainLabel, testLabel


if __name__ == '__main__':

	dataPath = '../imgAug'
	n_class = os.listdir(dataPath)
	label = 0
	imgsAll = []
	labelAll = []
	for count, className in enumerate(n_class):
		if not className == str('.DS_Store'):
			classPath = os.path.join(dataPath, className)
			imgList = os.listdir(classPath)
			for img in imgList:
				if not img == str('.DS_Store'):
					imgPath = os.path.join(classPath, img)
					imgsAll.append(imgPath)
					labelAll.append(label)

			label += 1

	trainImgs, testImgs, trainLabel, testLabel = split_trainTest(imgsAll, labelAll)

	createDataRecord('../data/train.tfrecords', trainImgs, trainLabel)
	createDataRecord('../data/test.tfrecords', testImgs, testLabel)

















