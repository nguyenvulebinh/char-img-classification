from collections import Counter

from PIL import Image
import numpy as np
import tensorflow as tf
from os import listdir
from os.path import isfile, join
import random

num_label = 62
data_path = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/data/"
filename_train = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/train.tfrecords"
filename_test = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/test.tfrecords"


################WRITE########################
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# images and labels array as input
def convert_to(image, label, writer):
    rows = image.shape[0]
    cols = image.shape[1]
    image_raw = image.tostring()
    one_hot_labels = np.zeros([num_label], dtype=np.int8)
    one_hot_labels[int(label)] = 1
    one_hot_labels = one_hot_labels.tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(rows),
        'width': _int64_feature(cols),
        'label': _bytes_feature(one_hot_labels),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    print("Write done.")


#####################Use to call write tfrecord###################
img_pare = []
for i in range(num_label):
    onlyfiles = [join(data_path + str(i), f) for f in listdir(data_path + str(i)) if
                 isfile(join(data_path + str(i), f))]
    for str_font_path in onlyfiles:
        img_pare.append((i, str_font_path))

random.shuffle(img_pare)
num_sample_test = int(len(img_pare) / 4)
num_sample_train = len(img_pare) - num_sample_test
print("All data: ", len(img_pare), "Train: ", num_sample_train, " Test: ", num_sample_test)

# File train
writer_train = tf.python_io.TFRecordWriter(filename_train)

# File test
writer_test = tf.python_io.TFRecordWriter(filename_test)

for label, img_path in img_pare:
    img = np.array(Image.open(img_path))
    # Write train data
    if num_sample_train > 0:
        print("Writing ", img_path, " -> datatrain")
        convert_to(img, label, writer_train)
        num_sample_train -= 1
    else:
        print("Writing ", img_path, " -> datatest")
        convert_to(img, label, writer_test)

writer_train.close()
writer_test.close()
