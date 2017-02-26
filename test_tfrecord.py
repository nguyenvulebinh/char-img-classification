import tensorflow as tf
import numpy as np
from PIL import Image

# The notMNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The notMNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CHANNEL = 1
count_test = np.zeros([NUM_CLASSES], dtype=np.int64)
count_train = np.zeros([NUM_CLASSES], dtype=np.int64)
filename_train = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/datatrain_not_number.tfrecords"
filename_test = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/datatest_not_number.tfrecords"
print_sample = 1

################READ########################
def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    _image = tf.decode_raw(features['image_raw'], tf.uint8)
    _image = tf.reshape(_image, [IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    _image.set_shape([IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNEL])
    _image = tf.to_float(_image)

    _label = tf.decode_raw(features['label'], tf.uint8)
    _label = tf.reshape(_label, [NUM_CLASSES])
    _label.set_shape([NUM_CLASSES])

    return _image, _label


with tf.Graph().as_default():
    image_test, label_test = read_and_decode(tf.train.string_input_producer([filename_test], num_epochs=1))
    images_batch_test, labels_batch_test = tf.train.shuffle_batch([image_test, label_test], batch_size=1,
                                                                  capacity=2,
                                                                  min_after_dequeue=1)

    image, label = read_and_decode(tf.train.string_input_producer([filename_train], num_epochs=1))
    images_batch, labels_batch = tf.train.shuffle_batch([image, label], batch_size=1, capacity=2,
                                                        min_after_dequeue=1)

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                images, labels = sess.run([images_batch, labels_batch])
                np.add(count_train, labels[0], count_train)
                print(labels[0])
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)
        sess.close()
    # Test session
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            while not coord.should_stop():
                images_test, labels_test = sess.run([images_batch_test, labels_batch_test])
                print(labels_test[0])
                np.add(count_test, labels_test[0], count_test)
        except tf.errors.OutOfRangeError:
            print('Done test -- epoch limit reached')
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()
        coord.join(threads)
        sess.close()

print(count_test)
print(count_train)
np.add(count_test, count_train, count_train)
print(count_train)
