import cv2
import numpy as np
import tensorflow as tf

OUTPUT_PATH = "../events/"


def with_update_op():
    input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32, name='input_node')
    training_node = tf.placeholder_with_default(True, (), name='training')

    net = tf.layers.conv2d(input_node, 32, (3, 3), strides=(2, 2), padding='same', name='conv_1')
    net = tf.layers.batch_normalization(net, training=training_node, name='bn')

    moving_mean = tf.get_default_graph().get_tensor_by_name(
        "bn/moving_mean/read:0")
    moving_var = tf.get_default_graph().get_tensor_by_name(
        "bn/moving_variance/read:0")

    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    print(f'update_op: {update_op}')

    with tf.control_dependencies(update_op):
        train_op = tf.identity(net, name='train_op')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        image = cv2.imread('../05/ithome.jpg')
        image = np.expand_dims(image, 0)

        for _ in range(100):
            sess.run(train_op, feed_dict={input_node: image})

        result, mm, mv = sess.run([net, moving_mean, moving_var], feed_dict={input_node: image, training_node: False})
        print(f'with_update_op:\n(mm , mv) : ({mm[0]:.2f} , {mv[0]:.2f})\n{result[0, 22:28, 22:28, 0]}')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def without_update_op():
    input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32, name='input_node')
    training_node = tf.placeholder_with_default(True, (), name='training')

    net = tf.layers.conv2d(input_node, 32, (3, 3), strides=(2, 2), padding='same', name='conv_1')
    net = tf.layers.batch_normalization(net, training=training_node, name='bn')

    moving_mean = tf.get_default_graph().get_tensor_by_name(
        "bn/moving_mean/read:0")
    moving_var = tf.get_default_graph().get_tensor_by_name(
        "bn/moving_variance/read:0")

    train_op = tf.identity(net, name='train_op')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        image = cv2.imread('../05/ithome.jpg')
        image = np.expand_dims(image, 0)

        for _ in range(10):
            sess.run(train_op, feed_dict={input_node: image})

        result, mm, mv = sess.run([net, moving_mean, moving_var], feed_dict={input_node: image, training_node: False})
        print(f'without_update_op:\n(mm , mv) : ({mm[0]:.2f} , {mv[0]:.2f})\n{result[0, 22:28, 22:28, 0]}')


if __name__ == '__main__':
    with_update_op()
    tf.reset_default_graph()
    without_update_op()
