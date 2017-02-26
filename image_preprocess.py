import os

import tensorflow as tf
from PIL import Image
from os import listdir
from os.path import isfile, join

image_default_size = 28
raw_image_folder = '/home/nguyenbinh/Programming/Python/PycharmWorkspace/MachineLearning/data'

# step 1
filenames = [join(raw_image_folder + os.sep + str(i), f) for i in range(62) for f in
             listdir(raw_image_folder + os.sep + str(i)) if
             isfile(join(raw_image_folder + os.sep + str(i), f))]

# step 2
filename_queue = tf.train.string_input_producer(filenames, shuffle=False, num_epochs=1)

# step 3: read, decode and resize images
reader = tf.WholeFileReader()
filename, content = reader.read(filename_queue)
image = tf.image.decode_png(content)
# zoom image
num_pixel_add = tf.random_shuffle([i for i in range(5, 10)])
image_zoom = tf.image.resize_image_with_crop_or_pad(
    tf.image.resize_images(image, [image_default_size + num_pixel_add[0], image_default_size + num_pixel_add[0]]),
    image_default_size, image_default_size)
# brightness image
image_brightness = tf.image.random_brightness(image, 0.5)
# contrast image
image_contrast = tf.image.random_contrast(image, 1, 5)
#Rotate image
degree_add = tf.random_shuffle([i for i in range(-20, 20)])
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
        while not coord.should_stop():
            # Prepare ndarray
            zoom, brightness, contrast, rotate, name_img, degree, pixel_add = sess.run(
                [image_zoom, image_brightness, image_contrast, image, filename, degree_add, num_pixel_add])
            zoom = zoom.reshape(image_default_size, image_default_size)
            brightness = brightness.reshape(image_default_size, image_default_size)
            contrast = contrast.reshape(image_default_size, image_default_size)
            rotate = rotate.reshape(image_default_size, image_default_size)
            name_img = name_img.decode("utf-8")
            degree = degree[0]
            print("Process ", name_img)
            # Write zoom image
            im = Image.fromarray(zoom).convert('L')
            im.save(name_img.replace(".png", "_zoom_"+str(pixel_add[0])+".png"))
            # Write brightness image
            im = Image.fromarray(brightness).convert('L')
            im.save(name_img.replace(".png", "_brightness.png"))
            # Write contrast image
            im = Image.fromarray(contrast).convert('L')
            im.save(name_img.replace(".png", "_contrast.png"))
            # Write rotate image
            im = Image.fromarray(rotate).convert('L')
            im = im.rotate(degree)
            im.save(name_img.replace(".png", "_rotate_"+str(degree)+".png"))
    except tf.errors.OutOfRangeError:
        print('Done preprocessed image')
    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    coord.join(threads)
    sess.close()
