import re

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import node_def_pb2, graph_pb2
from tensorflow.python.platform import gfile

OUTPUT_PATH = '../events/'
MODEL_PB = '../pb/frozen_shape.pb'


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../ckpt/model.ckpt-720.meta')
        saver.restore(sess, '../ckpt/model.ckpt-720')

        frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['final_dense/MatMul'])

    update_graph(frozen_gd)

    # opt start #

    preprocess_gd = add_preprocessing('backend/conv_1/Conv2D', 'input_node')
    update_graph(preprocess_gd)

    # opt end #

    with tf.Session() as sess:
        frozen_gd = tf.graph_util.convert_variables_to_constants(
            sess, preprocess_gd, ['final_dense/MatMul'])

    ensure_graph_is_valid(frozen_gd)

    tf.summary.FileWriter(OUTPUT_PATH, graph=frozen_gd)

    tf.io.write_graph(frozen_gd, "../pb/", "frozen_shape_24.pb", as_text=False)
    tf.io.write_graph(frozen_gd, "../pb/", "frozen_shape_24.pbtxt", as_text=True)


def update_graph(graph_def):
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')


def test():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile('../pb/frozen_shape_24.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_node = tf.get_default_graph().get_tensor_by_name(
            "new_input_node:0")
        training_node = tf.get_default_graph().get_tensor_by_name(
            "training:0")
        output_node = tf.get_default_graph().get_tensor_by_name(
            "final_dense/MatMul:0")

        image = cv2.imread('../05/ithome.jpg')
        image = cv2.resize(image, (128, 128))

        output = sess.run(output_node, feed_dict={
            input_node: np.expand_dims(image, 0),
            training_node: False})
        print(output)


def add_preprocessing(target_node_name, old_input_name):
    # create preprocessing node
    new_input_node = tf.placeholder(shape=[None, 128, 128, 3], dtype=tf.float32, name='new_input_node')
    with tf.variable_scope('pre_processing'):
        sub = tf.subtract(new_input_node, 127.5)
        mul = tf.multiply(sub, 0.0078125, name='out')

    old_gd = tf.get_default_graph().as_graph_def()
    old_nodes = old_gd.node  # old nodes from graph

    nodes_after_modify = []
    for node in old_nodes:
        new_node = node_def_pb2.NodeDef()  # 產生新節點
        new_node.CopyFrom(node)  # 拷貝舊節點資訊到新節點
        input_before_removal = node.input  # 把舊節點的inputs暫存起來
        if new_node.name == target_node_name:  # 如果節點是第一個con2D
            del new_node.input[:]  # 就把該inputs全部去除
            for input_name in input_before_removal:  # 然後再for跑一次剛剛刪除的inputs
                if input_name == old_input_name:  # inputs中若有舊input
                    new_node.input.append(mul.op.name)  # 指到新input
                else:
                    new_node.input.append(input_name)  # 不是的話，維持原先的input
        nodes_after_modify.append(new_node)  # 將新節點存到list

    new_gd = graph_pb2.GraphDef()  # 產生新graph def
    new_gd.node.extend(nodes_after_modify)  # 在新graph def中生成那些新節點後return
    return new_gd


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
