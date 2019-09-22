import glob
import os
import random
import shutil

import cv2
import numpy as np
import tensorflow as tf

OUTPUT_PATH = 'dataset/'
NUM_IMAGE = 960
SIZE = 128


def gen_data():
    shutil.rmtree(os.path.join(OUTPUT_PATH, 'line'), ignore_errors=True)
    shutil.rmtree(os.path.join(OUTPUT_PATH, 'rectangle'), ignore_errors=True)
    shutil.rmtree(os.path.join(OUTPUT_PATH, 'circle'), ignore_errors=True)

    os.mkdir(os.path.join(OUTPUT_PATH, 'line'))
    os.mkdir(os.path.join(OUTPUT_PATH, 'rectangle'))
    os.mkdir(os.path.join(OUTPUT_PATH, 'circle'))

    for idx in range(NUM_IMAGE):

        BG_B = random.randint(0, 255)
        BG_G = random.randint(0, 255)
        BG_R = random.randint(0, 255)

        OBJ_B = random.randint(0, 255)
        OBJ_G = random.randint(0, 255)
        OBJ_R = random.randint(0, 255)
        OBJ_DIFF = random.randint(0, 20)

        img = np.zeros((SIZE, SIZE, 3), np.uint8)

        img[:, :, 0] = BG_B
        img[:, :, 1] = BG_G
        img[:, :, 2] = BG_R

        type = random.randint(0, 2)

        if type % 3 == 0:
            cv2.line(img, (0 + OBJ_DIFF, 0 + OBJ_DIFF), (SIZE - OBJ_DIFF, SIZE - OBJ_DIFF),
                     (OBJ_B, OBJ_G, OBJ_R), 5)
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'line', f'{idx}.jpg'), img)
        elif type % 3 == 1:
            cv2.rectangle(img, (int(SIZE / 3 + OBJ_DIFF), int(SIZE / 3 + OBJ_DIFF)), (SIZE - OBJ_DIFF, SIZE - OBJ_DIFF),
                          (OBJ_B, OBJ_G, OBJ_R), 2)
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'rectangle', f'{idx}.jpg'), img)

        else:
            cv2.circle(img, (int(SIZE / 3 + OBJ_DIFF), int(SIZE / 3 + OBJ_DIFF)), 10 + OBJ_DIFF,
                       (OBJ_B, OBJ_G, OBJ_R), 3)
            cv2.imwrite(os.path.join(OUTPUT_PATH, 'circle', f'{idx}.jpg'), img)

    print('done.')


def gen_tfrecord():
    img_path_list = []
    img_path_list.extend(glob.glob('dataset/line/*.jpg'))
    img_path_list.extend(glob.glob('dataset/rectangle/*.jpg'))
    img_path_list.extend(glob.glob('dataset/circle/*.jpg'))
    random.shuffle(img_path_list)

    divide = int(NUM_IMAGE * 0.8)

    train_datas = img_path_list[:divide]
    test_datas = img_path_list[divide:]

    write_record('../tfrecord/train_shape.tfrecord', train_datas)
    write_record('../tfrecord/test_shape.tfrecord', test_datas)


def write_record(path, train_datas):
    writer = tf.python_io.TFRecordWriter(path)

    for data_path in train_datas:
        shape_type = data_path.split('/')[-2]
        if shape_type == 'line':
            type_index = 0
        elif shape_type == 'rectangle':
            type_index = 1
        else:
            type_index = 2
        with tf.gfile.GFile(os.path.join(data_path), 'rb') as fid:
            encoded_image = fid.read()

        tf_example = data_to_example(shape_type, type_index, encoded_image)
        writer.write(tf_example.SerializeToString())

    writer.close()


def data_to_example(shape_type, type_index, image):
    example = tf.train.Example(features=tf.train.Features(feature={
        'shape/type': bytes_feature(shape_type.encode('utf8')),
        'shape/type_index': int64_feature(type_index),
        'shape/image': bytes_feature(image),
    }))
    return example


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    gen_data()
    gen_tfrecord()
