import tensorflow as tf

OUTPUT_PATH = "../events/"


def main():
    input_node = tf.placeholder(shape=[None, 100, 100, 3], dtype=tf.float32)
    net = tf.layers.conv2d(input_node, 32, (3, 3), strides=(2, 2), padding='same', name='conv_1')
    net = tf.layers.conv2d(net, 32, (3, 3), strides=(1, 1), padding='same', name='conv_2')
    net = tf.layers.conv2d(net, 64, (3, 3), strides=(2, 2), padding='same', name='conv_3')

    tf.io.write_graph(tf.get_default_graph(), "../pb/", "model.pb", as_text=False)
    tf.io.write_graph(tf.get_default_graph(), "../pb/", "model.pbtxt", as_text=True)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph(), filename_suffix='_model_arch')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['conv_3/BiasAdd'])

        tf.io.write_graph(frozen_graph, "../pb/", "frozen_model.pb", as_text=False)
        tf.io.write_graph(frozen_graph, "../pb/", "frozen_model.pbtxt", as_text=True)

        tf.summary.FileWriter(OUTPUT_PATH, graph=frozen_graph, filename_suffix='_model_value')


if __name__ == '__main__':
    main()
