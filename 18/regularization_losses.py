import tensorflow as tf
import numpy as np

OUTPUT_PATH = '../events/'

INPUT_VALUE = np.asarray([[1, 2], [3, 4], [5, 6]])  # 3*2 的資料
WEIGHT_VALUE = np.asarray([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])  # 2*4 的權重
BIAS_VALUE = np.asarray([0.1, 0.2, 0.3, 0.4])  # bias = 4

REGULARIZER = tf.contrib.layers.l2_regularizer(0.1)


def raw_dense():
    x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x')

    w = tf.get_variable(
        name='weight',
        shape=(2, 4),
        dtype=tf.float32,
        regularizer=REGULARIZER)
    b = tf.get_variable(
        name='bias',
        shape=4,
        dtype=tf.float32,
        regularizer=REGULARIZER)

    assign_w = tf.assign(w, WEIGHT_VALUE)
    assign_b = tf.assign(b, BIAS_VALUE)

    out = tf.matmul(x, w) + b

    # tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run([assign_w, assign_b])

        result = sess.run(out, feed_dict={x: INPUT_VALUE})

        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        weight_loss = sess.run(wd_loss)

        print(f'result:{result}, weight_loss:{weight_loss}')

    return result, weight_loss


def layer_dense():
    x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x')

    weight_init = tf.constant_initializer(WEIGHT_VALUE)
    bias_init = tf.constant_initializer(BIAS_VALUE)
    out = tf.layers.dense(x, 4, kernel_initializer=weight_init, bias_initializer=bias_init,
                          kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER)

    # tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        result = sess.run(out, feed_dict={x: INPUT_VALUE})

        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        weight_loss = sess.run(wd_loss)

        print(f'result:{result}, weight_loss:{weight_loss}')

    return result, weight_loss


if __name__ == '__main__':
    raw_value, raw_lose = raw_dense()
    tf.reset_default_graph()
    layers_value, layers_loss = layer_dense()

    assert np.alltrue(raw_value == layers_value)
    assert raw_lose == layers_loss
