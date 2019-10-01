# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference_lib.py
import re

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

OUTPUT_PATH = '../events/'
MODEL_PB = '../pb/frozen_shape.pb'


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../ckpt/model.ckpt-720.meta')
        saver.restore(sess, '../ckpt/model.ckpt-720')

        # tf.summary.FileWriter(OUTPUT_PATH, graph=sess.graph)

        frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['final_dense/MatMul'])
        ensure_graph_is_valid(frozen_gd)

        tf.summary.FileWriter(OUTPUT_PATH, graph=frozen_gd)

        tf.io.write_graph(frozen_gd, "../pb/", "frozen_shape_23.pb", as_text=False)
        tf.io.write_graph(frozen_gd, "../pb/", "frozen_shape_23.pbtxt", as_text=True)


def test():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile('../pb/frozen_shape_23.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_node = tf.get_default_graph().get_tensor_by_name(
            "input_node:0")
        training_node = tf.get_default_graph().get_tensor_by_name(
            "training:0")
        output_node = tf.get_default_graph().get_tensor_by_name(
            "final_dense/MatMul:0")

        image = cv2.imread('../05/ithome.jpg')
        image = cv2.resize(image, (128, 128))
        image = image - 127.5
        image = image * 0.0078125

        output = sess.run(output_node, feed_dict={input_node: np.expand_dims(image, 0), training_node: False})
        print(output)


def ensure_graph_is_valid(graph_def):
    node_map = {}
    for node in graph_def.node:
        if node.name not in node_map:
            node_map[node.name] = node
        else:
            raise ValueError("Duplicate node names detected for ", node.name)
    for node in graph_def.node:
        for input_name in node.input:
            input_node_name = node_name_from_input(input_name)
            if input_node_name not in node_map:
                raise ValueError("Input for ", node.name, " not found: ",
                                 input_name)


def node_name_from_input(node_name):
    """Strips off ports and other decorations to get the underlying node name."""
    if node_name.startswith("^"):
        node_name = node_name[1:]
    m = re.search(r"(.*):\d+$", node_name)
    if m:
        node_name = m.group(1)
    return node_name


if __name__ == '__main__':
    main()
    tf.reset_default_graph()
    test()
