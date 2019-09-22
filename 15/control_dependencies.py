import tensorflow as tf

OUTPUT_PATH = "../events/"


def main():
    v = tf.get_variable(name="variable", shape=(), initializer=tf.constant_initializer(0))

    add_op = tf.assign_add(v, 3)

    with tf.control_dependencies([add_op]):
        mul_op = tf.multiply(v, 10)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        result = sess.run(mul_op)
        print(result)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


if __name__ == '__main__':
    main()
