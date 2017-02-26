import tensorflow as tf
import numpy as np

# The notMNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 62

# The notMNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_CHANNEL = 1
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

filename_train = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/test.tfrecords"
filename_test = "/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/tfrecord/datatest_all_clean.tfrecords"

folder_tensorboard = "notmnist_ckpt/sumary"
filename_checkpoint = "notmnist_ckpt/notminst.ckpt"
BATCH_SIZE = 128
CAPACITY = 400
MIN_AFTER_DEQUEUE = 100
NUM_EPOCHS = 10


#################Dinh nghia cac function###########################
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def inference(images____, drop_out):
    with tf.variable_scope('conv_1') as scope:
        kernel_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev=0.1))
        biases_1 = tf.Variable(tf.zeros([16]))
        conv_1 = tf.nn.conv2d(images____, kernel_1, [1, 1, 1, 1], padding='SAME')
        pre_activation_1 = tf.nn.bias_add(conv_1, biases_1)
        conv_1 = tf.nn.relu(pre_activation_1, name=scope.name)
    # variable_summaries(conv_1)
    # pool_1
    pool_1 = tf.nn.max_pool(conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool_1')

    with tf.variable_scope('conv_2') as scope:
        kernel_2 = tf.Variable(tf.truncated_normal([5, 5, 16, 16], stddev=0.1))
        biases_2 = tf.Variable(tf.zeros([16]))
        conv_2 = tf.nn.conv2d(pool_1, kernel_2, [1, 1, 1, 1], padding='SAME')
        pre_activation_2 = tf.nn.bias_add(conv_2, biases_2)
        conv_2 = tf.nn.relu(pre_activation_2, name=scope.name)
    # variable_summaries(conv_2)
    # pool_2
    pool_2 = tf.nn.max_pool(conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME', name='pool_2')

    # flatten_1
    flatten_1 = tf.reshape(pool_2, [-1, 16 * 7 * 7], name='flatten_1')

    # fullconnection_1
    with tf.variable_scope('fullconnection_1') as scope:
        fc_weights_1 = tf.Variable(tf.truncated_normal([16 * 7 * 7, 128], stddev=0.04))
        fc_biases_1 = tf.Variable(tf.zeros([128]))
        fc_1 = tf.nn.relu(tf.matmul(flatten_1, fc_weights_1) + fc_biases_1, name=scope.name)
    # variable_summaries(fc_1)
    # dropout_1
    dropout_1 = tf.nn.dropout(fc_1, drop_out, name="dropout_1")

    # fullconnection_2
    with tf.variable_scope('fullconnection_2') as scope:
        fc_weights_2 = tf.Variable(tf.truncated_normal([128, 64], stddev=0.04))
        fc_biases_2 = tf.Variable(tf.zeros([64]))
        fc_2 = tf.nn.relu(tf.matmul(dropout_1, fc_weights_2) + fc_biases_2, name=scope.name)
    # variable_summaries(fc_2)
    # dropout_2
    dropout_2 = tf.nn.dropout(fc_2, drop_out, name="dropout_2")

    # fullconnection_3
    with tf.variable_scope('fullconnection_3') as scope:
        fc_weights_3 = tf.Variable(tf.truncated_normal([64, NUM_CLASSES], stddev=0.04))
        fc_biases_3 = tf.Variable(tf.zeros([NUM_CLASSES]))
        fc_3 = tf.nn.relu(tf.matmul(dropout_2, fc_weights_3) + fc_biases_3, name=scope.name)
    # variable_summaries(fc_3)
    return fc_3


def loss(logits__, labels__):
    loss_val = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits__, labels=labels__))
    tf.summary.scalar('loss', loss_val)
    return loss_val


def training(loss___):
    return tf.train.AdamOptimizer(0.0000001).minimize(loss___)


def accuracy(predictions, labels__):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels__, 1))
            / predictions.shape[0])


###################################################################

################READ########################
def read_and_decode_tfrecord(filename_queue):
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
    _image = tf.reshape(_image, [IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    _image.set_shape([IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
    _image = tf.to_float(_image)

    _label = tf.decode_raw(features['label'], tf.uint8)
    _label = tf.reshape(_label, [NUM_CLASSES])
    _label.set_shape([NUM_CLASSES])

    return _image, _label


with tf.Graph().as_default():
    with tf.variable_scope("read_tfrecord"):
        # Defind input
        image_test_reader, label_test_reader = read_and_decode_tfrecord(
            tf.train.string_input_producer([filename_test], num_epochs=1))
        images_batch_test_op, labels_batch_test_op = tf.train.shuffle_batch([image_test_reader, label_test_reader],
                                                                            batch_size=BATCH_SIZE,
                                                                            capacity=CAPACITY,
                                                                            min_after_dequeue=MIN_AFTER_DEQUEUE)

        image_train_reader, label_train_reader = read_and_decode_tfrecord(
            tf.train.string_input_producer([filename_train], num_epochs=NUM_EPOCHS))
        images_batch_train_op, labels_batch_train_op = tf.train.shuffle_batch([image_train_reader, label_train_reader],
                                                                              batch_size=BATCH_SIZE, capacity=CAPACITY,
                                                                              min_after_dequeue=MIN_AFTER_DEQUEUE)

    with tf.variable_scope("input"):
        images_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL],
                                            name="placeholder_image")
        labels_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])
        drop_out_placeholder = tf.placeholder(dtype=tf.float32, shape=[], name="placeholder_dropout")

    with tf.variable_scope("train"):
        logistic_op = inference(images_placeholder, drop_out_placeholder)
        loss_op = loss(logistic_op, labels_placeholder)
        train_op = training(loss_op)
        global_step = tf.Variable(0, trainable=False, dtype=tf.uint16)
        is_checkpoint_exist = tf.train.checkpoint_exists(filename_checkpoint)

    with tf.variable_scope("output"):
        predictions_raw = tf.nn.softmax(logistic_op, name="predictions")

    with tf.variable_scope("sumary"):
        merged = tf.summary.merge_all()

    saver = tf.train.Saver()

    # # train session
    # with tf.Session() as sess:
    #     sess.run(tf.local_variables_initializer())
    #     sess.run(tf.global_variables_initializer())
    #     summary_writer = tf.summary.FileWriter(folder_tensorboard, sess.graph)
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    #     if is_checkpoint_exist:
    #         saver.restore(sess, filename_checkpoint)
    #     try:
    #         while not coord.should_stop():
    #             images_train, labels_train, step = sess.run(
    #                 [images_batch_train_op, labels_batch_train_op, global_step.assign_add(1)])
    #
    #             _, loss = sess.run(
    #                 [train_op, loss_op],
    #                 feed_dict={images_placeholder: images_train,
    #                            labels_placeholder: labels_train,
    #                            drop_out_placeholder: 0.85})
    #             print("Step: ", step, " Loss: ", loss)
    #
    #             if step % 10 == 0:
    #                 predictions_train, merged_str = sess.run(
    #                     [predictions_raw, merged],
    #                     feed_dict={images_placeholder: images_train,
    #                                labels_placeholder: labels_train,
    #                                drop_out_placeholder: 1})
    #
    #                 true_batch = np.sum(np.argmax(predictions_train, 1) == np.argmax(labels_train, 1))
    #                 print("Step: ", step, " Minibatch accuracy: ",
    #                       (true_batch * 1.0 / images_train.shape[0]) * 100)
    #
    #                 summary_writer.add_summary(merged_str, step)
    #                 summary_writer.flush()
    #                 saver.save(sess, filename_checkpoint)
    #
    #             step += 1
    #
    #     except tf.errors.OutOfRangeError:
    #         print('Done training -- epoch limit reached')
    #     finally:
    #         # When done, ask the threads to stop.
    #         coord.request_stop()
    #         summary_writer.close()
    #
    #     coord.join(threads)
    #     sess.close()

    # Test session
    with tf.Session() as sess:
        count_all = 0
        count_true = 0
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver.restore(sess, filename_checkpoint)
        try:
            while not coord.should_stop():
                images_test, labels_test = sess.run([images_batch_test_op, labels_batch_test_op])
                predictions_test = sess.run(predictions_raw,
                                            feed_dict={images_placeholder: images_test,
                                                       labels_placeholder: labels_test,
                                                       drop_out_placeholder: 1})
                count_true += np.sum(np.argmax(predictions_test, 1) == np.argmax(labels_test, 1))
                count_all += images_test.shape[0]
                print(count_true, count_all)
        except tf.errors.OutOfRangeError:
            print('Done test -- epoch limit reached')
            print(count_true, count_all, "Accuracy: ", (count_true * 1.0 / count_all) * 100)
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        coord.join(threads)
        sess.close()
