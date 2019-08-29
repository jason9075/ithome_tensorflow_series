import cv2
import tensorflow as tf
import numpy as np

OUTPUT_PATH = "../events/"
NUM_FILTERS = 10
FILTER_SIZE = (3, 3)
STRIDES = (1, 1)


def nn(input_node):
    with tf.variable_scope('nn'):
        w = tf.get_variable(
            name='weight',
            shape=[FILTER_SIZE[0], FILTER_SIZE[1], 3, NUM_FILTERS],
            dtype=tf.float32)
        b = tf.get_variable(
            name='bias',
            shape=[NUM_FILTERS],
            dtype=tf.float32)
        out = tf.nn.conv2d(input_node, filter=w, strides=(1, 1),
                           padding='SAME')
        out = out + b

    return out


def layer(input_node):
    out = tf.layers.conv2d(input_node, NUM_FILTERS, FILTER_SIZE, strides=STRIDES, padding='same', name='layer')

    return out


def slim(input_node):
    out = tf.contrib.slim.conv2d(input_node, NUM_FILTERS, FILTER_SIZE, stride=STRIDES, padding='SAME',
                                 activation_fn=None, scope='slim')

    return out


def keras(input_node):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(NUM_FILTERS, FILTER_SIZE, strides=STRIDES, padding='same')
    ], name='keras')
    return model(input_node)


if __name__ == '__main__':
    node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)

    nn_out = nn(node)
    layer_out = layer(node)
    slim_out = slim(node)
    keras_out = keras(node)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())

    image = cv2.imread('ithome.jpg')
    image = np.expand_dims(image, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        nn_result, layer_result, slim_result, keras_result = \
            sess.run([nn_out, layer_out, slim_out, keras_out], feed_dict={node: image})

        print(f'nn shape: {nn_result.shape}')
        print(f'layer shape: {layer_result.shape}')
        print(f'slim shape: {slim_result.shape}')
        print(f'keras shape: {keras_result.shape}')
