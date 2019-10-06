import timeit

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

TIMES = 1000


def before_opt():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile('../pb/frozen_shape_23.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_node = tf.get_default_graph().get_tensor_by_name(
            "input_node:0")
        training_node = tf.get_default_graph().get_tensor_by_name(
            "training:0")
        output_node = tf.get_default_graph().get_tensor_by_name(
            "final_dense/MatMul:0")

        image = get_preprocess_image()
        output = sess.run(output_node, feed_dict={
            input_node: np.expand_dims(image, 0),
            training_node: False})

        start = timeit.default_timer()
        for _ in range(0, TIMES):
            image = get_preprocess_image()
            sess.run(output_node, feed_dict={
                input_node: np.expand_dims(image, 0),
                training_node: False})

        print(f'before opt cost time:{(timeit.default_timer() - start)} sec')
        print(f'output:{output}.\n')


def after_opt():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile('../pb/frozen_shape_28.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_node = tf.get_default_graph().get_tensor_by_name(
            "new_input_node:0")
        output_node = tf.get_default_graph().get_tensor_by_name(
            "final_dense/MatMul:0")

        image = get_row_image()
        output = sess.run(output_node, feed_dict={
            input_node: np.expand_dims(image, 0)})

        start = timeit.default_timer()
        for _ in range(0, TIMES):
            image = get_row_image()
            sess.run(output_node, feed_dict={
                input_node: np.expand_dims(image, 0)})

        print(f'after opt cost time:{(timeit.default_timer() - start)} sec')
        print(f'output:{output}.\n')


def get_preprocess_image():
    image = cv2.imread('../05/ithome.jpg')
    image = cv2.resize(image, (128, 128))
    image = image - 127.5
    image = image * 0.0078125
    return image


def get_row_image():
    image = cv2.imread('../05/ithome.jpg')
    image = cv2.resize(image, (128, 128))
    return image


if __name__ == '__main__':
    tf.reset_default_graph()
    before_opt()
    tf.reset_default_graph()
    after_opt()
