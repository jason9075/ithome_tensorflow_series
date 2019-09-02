import tensorflow as tf

OUTPUT_PATH = '../events/'


def main():
    input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
    training_node = tf.placeholder(shape=(), dtype=tf.bool)

    net = tf.layers.conv2d(input_node, 32, (3, 3), strides=(2, 2), padding='same', name='conv_1')
    net = tf.layers.batch_normalization(net, training=training_node, name='bn')

    net = tf.layers.conv2d(net, 32, (3, 3), strides=(1, 1), padding='same', name='conv_2')
    net = tf.layers.dropout(net, rate=0.6, training=training_node, name='dropout')

    tf.identity(net, name='final')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


if __name__ == '__main__':
    main()
