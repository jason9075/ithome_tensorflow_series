import tensorflow as tf


def main():

    tf.constant(0, name="c0")

    with tf.Graph().as_default() as g1:
        tf.constant(1, name="c1")

    with tf.Graph().as_default() as g2:
        tf.constant(2, name="c2")

    sess0 = tf.Session()
    sess1 = tf.Session(graph=g1)
    sess2 = tf.Session(graph=g2)

    t0 = tf.get_default_graph().get_tensor_by_name('c0:0')
    t1 = sess1.graph.get_tensor_by_name('c1:0')
    t2 = sess2.graph.get_tensor_by_name('c2:0')

    result0 = sess0.run(t0)
    result1 = sess1.run(t1)
    result2 = sess2.run(t2)

    print("result 0: {}".format(result0))
    print("result 1: {}".format(result1))
    print("result 2: {}".format(result2))

    print("default graph: {}".format([n.name for n in tf.get_default_graph().as_graph_def().node]))
    print("graph 1: {}".format([n.name for n in g1.as_graph_def().node]))
    print("graph 2: {}".format([n.name for n in g2.as_graph_def().node]))

    sess0.close()
    sess1.close()
    sess2.close()


if __name__ == '__main__':
    main()
