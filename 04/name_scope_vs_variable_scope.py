import tensorflow as tf

OUTPUT_PATH = "../events/"


def ex1():
    with tf.name_scope('name'):
        tf.placeholder(tf.int32, name='i1')

    with tf.variable_scope('variable'):
        tf.placeholder(tf.int32, name='i2')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def ex2():
    tf.get_variable(name="V", shape=[1])
    # tf.get_variable(name="V", shape=[1])  # it will crash
    tf.Variable(name="V", initial_value=0)
    tf.Variable(name="V", initial_value=0)

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def ex3():
    with tf.name_scope('name'):
        tf.get_variable(name="V", shape=[1])

    with tf.variable_scope('variable'):
        tf.get_variable(name="V", shape=[1])

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def ex4():
    with tf.variable_scope("foo"):
        v = tf.get_variable("v", [1])

    # 其他程式
    # ...
    # ...
    # ...
    # ...

    with tf.variable_scope("foo", reuse=True):
        v1 = tf.get_variable("v", [1])

    assert v1 == v

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def ex5():
    with tf.variable_scope("foo") as scope:
        v = tf.get_variable("v", [1])

        # 其他程式
        # ...
        # ...
        # ...
        # ...

        scope.reuse_variables()
        v1 = tf.get_variable("v", [1])

    assert v1 == v

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


if __name__ == '__main__':
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    ex5()
