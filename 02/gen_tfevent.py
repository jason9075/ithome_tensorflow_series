import tensorflow as tf

OUTPUT_PATH = "../events/"


def main():
    input_node = tf.placeholder(tf.int32)
    constant = tf.constant(1, tf.int32, name='constant')

    tf.add(input_node, constant, name='result')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())

    print('done.')


if __name__ == '__main__':
    main()
