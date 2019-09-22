import tensorflow as tf

OUTPUT_PATH = "../events/"
EPOCHS = 25


def main():

    global_step = tf.train.get_or_create_global_step()
    update_op = tf.assign_add(global_step, 1)

    lr_steps = [5, 10, 15]
    lr_values = [0.1, 0.05, 0.01, 0.001]

    with tf.control_dependencies([update_op]):
        lr_value = tf.train.piecewise_constant(global_step, boundaries=lr_steps, values=lr_values, name='lr_rate')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for _ in range(EPOCHS):
            gs, lr = sess.run([global_step, lr_value])

            print(f'step {gs}, lr = {lr:.3f}')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


if __name__ == '__main__':
    main()
