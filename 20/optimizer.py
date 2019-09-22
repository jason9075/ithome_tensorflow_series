import numpy as np
import tensorflow as tf
import timeit

OUTPUT_PATH = '../events/'

BIAS_INIT = tf.constant_initializer(0.0)
WEIGHT_INIT = tf.truncated_normal_initializer(stddev=1.0)
REGULARIZER = tf.contrib.layers.l2_regularizer(0.1)


def main(opt_type):
    global_step = tf.get_variable(name="global_step", shape=(), initializer=tf.constant_initializer(0),
                                  dtype=tf.int32, trainable=False)

    x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None], dtype=tf.int32, name='y')

    with tf.variable_scope('backend'):
        net = tf.layers.dense(x, 64, activation=tf.nn.relu6, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                              kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='dense_1')
        net = tf.layers.dense(net, 64, activation=tf.nn.relu6, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                              kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='dense_2')
        logits = tf.layers.dense(net, 2, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                                 kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='final_dense')

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name='inference_loss')

    opt = get_optimizer(opt_type)

    grads = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        start = timeit.default_timer()
        for _ in range(0, 100):
            x_v, y_v = get_xor_data()
            _, count = sess.run([train_op, global_step], feed_dict={x: x_v, y: y_v})

            print(f'iter: {count}')

        print(f'done. cost {timeit.default_timer() - start} sec.')

        tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def get_optimizer(opt_type):
    if opt_type == 'gd':
        return tf.train.GradientDescentOptimizer(learning_rate=0.1)
    if opt_type == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)
    if opt_type == 'ada_grad':
        return tf.train.AdagradOptimizer(learning_rate=0.1, initial_accumulator_value=0.1)
    if opt_type == 'rmsp':
        return tf.train.RMSPropOptimizer(learning_rate=0.1, decay=0.9, momentum=0)
    if opt_type == 'adam':
        return tf.train.AdamOptimizer(learning_rate=0.1, beta1=0.9, beta2=0.99)


def get_xor_data():
    x = (np.random.rand(16, 2) - 0.5) * 2
    y = [0 if 0 < x1 * x2 else 1 for x1, x2 in x]

    return x, y


if __name__ == '__main__':
    main('adam')
