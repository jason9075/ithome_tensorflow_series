import cv2
import numpy as np
import tensorflow as tf
import timeit

OUTPUT_PATH = '../events/'

NUM_CLASSES = 10


def alexnet():
    input_node = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32, name='input_node')

    net = tf.layers.conv2d(input_node, 96, (11, 11), strides=(4, 4), activation=tf.nn.relu,
                           padding='same', name='conv_1')  # 55x55
    net = tf.nn.lrn(net, depth_radius=5, bias=1.0, alpha=0.0001 / 5.0, beta=0.75, name='norm_1')
    net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=2, name='max_pool_1')  # 27x27

    net = tf.layers.conv2d(net, 256, (5, 5), strides=(1, 1), activation=tf.nn.relu,
                           padding="same", name='conv_2')
    net = tf.nn.lrn(net, depth_radius=5, bias=1.0, alpha=0.0001 / 5.0, beta=0.75, name='norm_2')
    net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=2, name='max_pool_2')  # 13x13

    net = tf.layers.conv2d(net, 384, (3, 3), strides=(1, 1), padding="same", activation=tf.nn.relu, name='conv_3')

    net = tf.layers.conv2d(net, 384, (3, 3), strides=(1, 1), padding="same", activation=tf.nn.relu, name='conv_4')

    net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), padding="same", activation=tf.nn.relu, name='conv_5')
    net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=2, padding="valid", name='max_pool_5')  # 6x6

    net = tf.reshape(net, [-1, 6 * 6 * 256], name='flat')
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense_1')

    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense_2')

    logits = tf.layers.dense(net, NUM_CLASSES, name="logits_layer")

    print_para_count()
    test_speed(input_node, logits)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def vgg16net():
    input_node = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32, name='input_node')

    net = tf.layers.conv2d(input_node, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_1_1')
    net = tf.layers.conv2d(net, 64, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_1_2')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2, name='max_pool_1')  # 112x112

    net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_2_1')
    net = tf.layers.conv2d(net, 128, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_2_2')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2, name='max_pool_2')  # 56x56

    net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_3_1')
    net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_3_2')
    net = tf.layers.conv2d(net, 256, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_3_3')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2, name='max_pool_3')  # 28x28

    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_4_1')
    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_4_2')
    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_4_3')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2, name='max_pool_4')  # 14x14

    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_5_1')
    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_5_2')
    net = tf.layers.conv2d(net, 512, (3, 3), strides=(1, 1), activation=tf.nn.relu,
                           padding='same', name='conv_5_3')
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=2, name='max_pool_5')  # 7x7

    net = tf.reshape(net, [-1, 7 * 7 * 512], name='flat')
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense_1')
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense_2')
    net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='dense_3')
    logits = tf.layers.dense(net, NUM_CLASSES, name='logits_layer')

    print_para_count()
    test_speed(input_node, logits)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def print_para_count():
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print(f'trainable parameters count: {total_parameters}')


def test_speed(input_node, logits):
    TIMES = 10
    image = cv2.imread('../05/ithome.jpg')
    image = cv2.resize(image, (input_node.shape[1], input_node.shape[2]))
    image = np.expand_dims(image, 0)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        sess.run(logits, feed_dict={input_node: image})  # <-- first time is slow

        start = timeit.default_timer()
        for _ in range(0, TIMES):
            sess.run(logits, feed_dict={input_node: image})

        print(f'cost time:{(timeit.default_timer() - start)} sec')


if __name__ == '__main__':
    alexnet()
    tf.reset_default_graph()
    vgg16net()
