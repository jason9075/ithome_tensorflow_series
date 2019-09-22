import tensorflow as tf
import numpy as np

OUTPUT_PATH = '../events/'

BIAS_INIT = tf.constant_initializer(0.0)
WEIGHT_INIT = tf.truncated_normal_initializer(stddev=1.0)
REGULARIZER = tf.contrib.layers.l2_regularizer(0.1)


def main():
    global_step = tf.train.get_or_create_global_step()
    x = tf.placeholder(shape=[None, 2], dtype=tf.float32, name='x')
    y = tf.placeholder(shape=[None], dtype=tf.int32, name='y')
    net = tf.layers.dense(x, 64, activation=tf.nn.relu6, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                          kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='dense_1')
    net = tf.layers.dense(net, 64, activation=tf.nn.relu6, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                          kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='dense_2')
    logits = tf.layers.dense(net, 2, kernel_initializer=WEIGHT_INIT, bias_initializer=BIAS_INIT,
                             kernel_regularizer=REGULARIZER, bias_regularizer=REGULARIZER, name='final_dense')

    inference_loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name='inference_loss')
    wd_loss = tf.reduce_sum(
        tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
    total_loss = tf.add(inference_loss, wd_loss, name='total_loss')
    acc, acc_op = tf.metrics.accuracy(y, tf.argmax(logits, 1), name='accuracy')

    opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    grads = opt.compute_gradients(total_loss)
    train_op = opt.apply_gradients(grads, global_step=global_step)

    summary = tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())
    summaries = []

    for grad, var in grads:
        if grad is not None:
            summaries.append(
                tf.summary.histogram(var.op.name + '/gradients', grad))

    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    summaries.append(tf.summary.scalar('inference_loss', inference_loss))
    summaries.append(tf.summary.scalar('wd_loss', wd_loss))
    summaries.append(tf.summary.scalar('total_loss', total_loss))
    summaries.append(tf.summary.scalar('accuracy', acc))
    summary_op = tf.summary.merge(summaries)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for _ in range(0, 100):
            x_v, y_v = get_xor_data()
            _, _, summary_op_val, step = sess.run(
                [train_op, acc_op, summary_op, global_step],
                feed_dict={x: x_v, y: y_v})

            summary.add_summary(summary_op_val, step)
            print(f'iter: {step}')

    summary.close()


def get_xor_data():
    x = (np.random.rand(16, 2) - 0.5) * 2
    y = [0 if 0 < x1 * x2 else 1 for x1, x2 in x]

    return x, y


if __name__ == '__main__':
    main()
