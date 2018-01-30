import tensorflow as tf
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

SAVE_PATH = './dataset.tfrecords'
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 650
root_path = r'F:\kaggle_data'

def _file_path(root_path):
    image_file = sorted(os.listdir(os.path.join(root_path,
                                                "image")))[:NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN]
    data_path = []
    for f in image_file:
        image_path = os.path.join(os.path.join(root_path, 'image'), f)
        mask_path = os.path.join(os.path.join(root_path, 'mask'), f)
        boundary_path = os.path.join(os.path.join(root_path, 'boundary'), f)
        path = [image_path, mask_path, boundary_path]
        data_path.append(path)

    return data_path

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def create_data(root_path):
    data_path = _file_path(root_path)

    writer = tf.python_io.TFRecordWriter(SAVE_PATH)

    for li in data_path:
        image_path = li[0]
        mask_path = li[1]
        boundary_path = li[2]

        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        image_raw = image.tostring()

        mask = cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)
        mask_raw = mask.tostring()

        boundary = cv2.imread(boundary_path, cv2.COLOR_BGR2GRAY)
        width, height = boundary.shape
        boundary_raw = boundary.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image':_bytes_feature(image_raw),
            'mask':_bytes_feature(mask_raw),
            'boundary':_bytes_feature(boundary_raw),
            'image_width':_int64_feature(width),
            'image_height':_int64_feature(height)
        }))

        writer.write(example.SerializeToString())
    writer.close()
    print('Dataset file create successfully!')

def _generator_dataset_and_label_batch(image, mask, boundary,
                                       min_queue_examples, batch_size, shuffle=True):
    num_preprocess_threads = 4
    if shuffle:
        images, masks, boundaries = tf.train.shuffle_batch(
            [image, mask, boundary],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, masks, boundaries = tf.train.batch(
            [image, mask, boundary],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # Display the training images in the visualizer.
    # tf.summary.image('images', images)
    # tf.summary.image('mask', mask)
    # tf.summary.image('bounday', boundary)

    return images, masks, boundaries

def load_batch(batch_size, WIDTH, HEIGHT, shuffle):
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([SAVE_PATH])

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={'image': tf.FixedLenFeature([], tf.string),
                  'mask': tf.FixedLenFeature([], tf.string),
                  'boundary': tf.FixedLenFeature([], tf.string),
                  'image_width': tf.FixedLenFeature([], tf.int64),
                  'image_height': tf.FixedLenFeature([], tf.int64)})

    image = tf.decode_raw(features['image'], tf.uint8)
    mask = tf.decode_raw(features['mask'], tf.uint8)
    boundary = tf.decode_raw(features['boundary'], tf.uint8)
    image_width = tf.cast(features['image_width'], tf.int32)
    image_height = tf.cast(features['image_height'], tf.int32)

    image = tf.reshape(image, [image_width, image_height, 3])
    mask = tf.reshape(mask, [image_width, image_height, 1])
    boundary = tf.reshape(boundary, [image_width, image_height, 1])

    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)
    boundary = tf.cast(boundary, tf.float32)

    seed_value = np.random.randint(0, 65535)
    image = tf.random_crop(image, [WIDTH, HEIGHT, 3], seed=seed_value,name='image_shape')
    mask = tf.random_crop(mask, [WIDTH, HEIGHT, 1], seed=seed_value, name='mask_shape')
    boundary = tf.random_crop(boundary, [WIDTH, HEIGHT, 1], seed=seed_value, name='boundary_shape')

    image = tf.image.random_flip_left_right(image, seed=seed_value)
    mask = tf.image.random_flip_left_right(mask, seed=seed_value)
    boundary = tf.image.random_flip_left_right(boundary, seed=seed_value)

    image = tf.image.random_flip_up_down(image, seed=seed_value)
    mask = tf.image.random_flip_up_down(mask, seed=seed_value)
    boundary = tf.image.random_flip_up_down(boundary, seed=seed_value)

    image = tf.subtract(tf.divide(image, 127.5), 1.0)
    mask = tf.subtract(tf.divide(mask, 127.5), 1.0)
    boundary = tf.subtract(tf.divide(boundary, 127.5), 1.0)

    image.set_shape([WIDTH, HEIGHT, 3])
    mask.set_shape([WIDTH, HEIGHT, 1])
    boundary.set_shape([WIDTH, HEIGHT, 1])

    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)

    return _generator_dataset_and_label_batch(image, mask, boundary, min_queue_examples, batch_size, shuffle=shuffle)

#test
if __name__ == '__main__':
    if not os.path.isfile(SAVE_PATH):
        create_data(root_path=root_path)
    images, masks, boundaries = load_batch(4, 256, 256, True)
    with tf.Session().as_default() as sess:

        init_op = tf.initialize_all_variables()
        init_op1 = tf.local_variables_initializer()
        sess.run(init_op)
        sess.run(init_op1)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            for i in range(15000):
                print(i)
                Images = sess.run(images)
                # print(Images[0])
                cv2.imshow('image',Images[0])
                # cv2.imshow('Mask', Masks[0])
                # cv2.imshow('Boundaries', Boundaries[0])
                cv2.waitKey(25)
        except tf.errors.OutOfRangeError:
            print('Done training for %d steps.' % (i))
        finally:
            coord.request_stop()
            coord.join(threads)
