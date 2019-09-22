import numpy as np
import tensorflow as tf

OUTPUT_PATH = '../events/'
TRAIN_TFRECORD_PATH = '../tfrecord/train_shape.tfrecord'
TEST_TFRECORD_PATH = '../tfrecord/test_shape.tfrecord'

BIAS_INIT = tf.constant_initializer(0.0)
WEIGHT_INIT = tf.truncated_normal_initializer(stddev=0.05)
REGULARIZER = tf.contrib.layers.l2_regularizer(0.01)

EPOCH = 30
BATCH_SIZE = 32


def main():
    # Build Model #
    global_step = tf.train.get_or_create_global_step()
    input_node = tf.placeholder(shape=[None, 128, 128, 3], dtype=tf.float32, name='input_node')
    training_node = tf.placeholder_with_default(True, shape=(), name='training')
    labels = tf.placeholder(shape=[None], dtype=tf.int64, name='img_labels')

    with tf.variable_scope('backend'):
        net = tf.layers.conv2d(input_node, 32, (3, 3),
                               activation=tf.nn.relu6,
                               strides=(1, 1),
                               padding='same',
                               name='conv_1')
        net = tf.layers.batch_normalization(net,
                                            training=training_node,
                                            name='bn_1')
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='max_pool_1')  # 64

        net = tf.layers.conv2d(net, 64, (3, 3),
                               activation=tf.nn.relu6,
                               strides=(1, 1),
                               padding='same',
                               name='conv_2')
        net = tf.layers.batch_normalization(net,
                                            training=training_node,
                                            name='bn_2')
        net = tf.layers.dropout(net, 0.1, training=training_node, name='dropout_2')
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='max_pool_2')  # 32

        net = tf.layers.conv2d(net, 128, (3, 3),
                               activation=tf.nn.relu6,
                               strides=(2, 2),
                               padding='same',
                               name='conv_3')  # 16
        net = tf.layers.batch_normalization(net,
                                            training=training_node,
                                            name='bn_3')
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), name='max_pool_3')  # 8

    net = tf.reshape(net, [-1, 8 * 8 * 128], name='flatten')
    logit = tf.layers.dense(net, 3, use_bias=False, kernel_initializer=WEIGHT_INIT,
                            kernel_regularizer=REGULARIZER, name='final_dense')

    lr = tf.train.piecewise_constant(global_step,
                                     boundaries=[250, 500, 750],
                                     values=[0.1, 0.05, 0.01, 0.001],
                                     name='lr_schedule')

    # Loss Function #
    with tf.variable_scope('loss'):
        inference_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logit, labels=labels), name='inference_loss')
        wd_loss = tf.reduce_sum(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='wd_loss')
        total_loss = tf.add(inference_loss, wd_loss, name='total_loss')
    train_acc, acc_op = tf.metrics.accuracy(labels, tf.argmax(logit, 1), name='accuracy')
    test_acc_node = tf.placeholder(dtype=tf.float32, shape=(), name='test_acc')

    # Optimizer #
    opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
    grads = opt.compute_gradients(total_loss)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(grads, global_step=global_step)

    # Summary #
    summary = tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())
    summaries = []

    for grad, var in grads:
        if grad is not None:
            summaries.append(
                tf.summary.histogram(var.op.name + '/gradients', grad))

    for var in tf.trainable_variables():
        summaries.append(tf.summary.histogram(var.op.name, var))

    summaries.append(tf.summary.scalar('loss/total', total_loss))
    summaries.append(tf.summary.scalar('loss/inference', inference_loss))
    summaries.append(tf.summary.scalar('loss/weight', wd_loss))
    summaries.append(tf.summary.scalar('accuracy/train', train_acc))
    summaries.append(tf.summary.scalar('accuracy/test', test_acc_node))
    summary_op = tf.summary.merge(summaries)

    # Parameters Count #
    total_parameters = 0
    for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    print('trainable parameters count: %d' % total_parameters)

    # Data Set #
    with tf.variable_scope('train_iterator'):
        data_set = tf.data.TFRecordDataset(TRAIN_TFRECORD_PATH)
        data_set = data_set.map(parse_function)
        data_set = data_set.batch(BATCH_SIZE)
        train_iterator = data_set.make_initializable_iterator()
        next_train_element = train_iterator.get_next()

    with tf.variable_scope('test_iterator'):
        data_set = tf.data.TFRecordDataset(TEST_TFRECORD_PATH)
        data_set = data_set.map(parse_function)
        data_set = data_set.batch(BATCH_SIZE)
        test_iterator = data_set.make_initializable_iterator()
        next_test_element = test_iterator.get_next()

    # Training #
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for _ in range(EPOCH):
            sess.run(train_iterator.initializer)

            while True:
                try:
                    labels_train, images_train = sess.run(next_train_element)
                    if images_train.shape[0] != BATCH_SIZE:
                        break

                    _, total_loss_val, train_acc_val, current_step = \
                        sess.run([train_op, total_loss, acc_op, global_step], feed_dict={input_node: images_train,
                                                                                         labels: labels_train})
                    print('step: %d, total_lost: %.2f, train_acc_val: %.2f' %
                          (current_step, total_loss_val, train_acc_val))

                    if current_step % 20 == 0:
                        test_acc = eval_acc(sess, test_iterator, next_test_element, logit, input_node, labels, training_node)
                        summary_op_val = \
                            sess.run(summary_op, feed_dict={input_node: images_train,
                                                            labels: labels_train,
                                                            test_acc_node: test_acc,
                                                            training_node: False})
                        print('test_acc: %.2f' % test_acc)
                        summary.add_summary(summary_op_val, current_step)

                except tf.errors.OutOfRangeError:
                    print('next epoch.')
                    break  # epoch end

        # Save to .ckpt #
        saver.save(sess, "../ckpt/model.ckpt", global_step=global_step, latest_filename='shape_model')


def eval_acc(sess, iterator, next_test_element, logit_op, input_node, labels, training_node):
    sess.run(iterator.initializer)
    correct = 0
    total = 0
    while True:
        try:
            labels_test, images_test = sess.run(next_test_element)
            if images_test.shape[0] != BATCH_SIZE:
                break
            logit = sess.run(logit_op,
                             feed_dict={input_node: images_test, labels: labels_test, training_node: False})
            prediction = np.argmax(logit, axis=1)

            comparison = [label == predict for label, predict in zip(labels_test, prediction)]
            correct += sum(comparison)
            total += len(comparison)

        except tf.errors.OutOfRangeError:
            break

    return correct / total


def parse_function(example_proto):
    features = {'shape/type': tf.io.FixedLenFeature([], tf.string),
                'shape/type_index': tf.io.FixedLenFeature([], tf.int64),
                'shape/image': tf.io.FixedLenFeature([], tf.string)}
    features = tf.io.parse_single_example(example_proto, features)
    images = tf.image.decode_jpeg(features['shape/image'], channels=3)
    images = tf.cast(images, dtype=tf.float32)
    images = tf.subtract(images, 127.5)
    images = tf.multiply(images, 0.0078125)

    return features['shape/type_index'], images


if __name__ == '__main__':
    main()
