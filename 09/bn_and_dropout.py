import tensorflow as tf

OUTPUT_PATH = '../events/'


def main():
    input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32, name='input_node')
    training_node = tf.placeholder(shape=(), dtype=tf.bool, name='training')

    net = tf.layers.conv2d(input_node, 32, (3, 3), strides=(2, 2), padding='same', name='conv_1')
    net = tf.layers.batch_normalization(net, training=training_node, name='bn')

    net = tf.layers.conv2d(net, 32, (3, 3), strides=(1, 1), padding='same', name='conv_2')
    net = tf.layers.dropout(net, rate=0.6, training=training_node, name='dropout')

    tf.identity(net, name='final')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['final'])

        tf.summary.FileWriter(OUTPUT_PATH, graph=frozen_graph)
        tf.io.write_graph(frozen_graph, "../pb/", "bn_dropout_model.pb", as_text=False)


if __name__ == '__main__':
    main()
