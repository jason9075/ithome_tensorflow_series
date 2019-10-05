import math
import re

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import gfile

OUTPUT_PATH = '../events/'
MODEL_PB = '../pb/frozen_shape.pb'


def main():
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('../ckpt/model.ckpt-720.meta')
        saver.restore(sess, '../ckpt/model.ckpt-720')

        frozen_graph = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['final_dense/MatMul'])

    update_graph(frozen_graph)

    # opt start #

    preprocess_gd = add_preprocessing('backend/conv_1/Conv2D', 'input_node')
    update_graph(preprocess_gd)

    no_training_gd = remove_training_node('training', 'false_node')
    update_graph(no_training_gd)

    no_dropout_gd = remove_dropout()
    update_graph(no_dropout_gd)

    no_merge_gd = remove_merge()
    update_graph(no_merge_gd)

    no_switch_gd = remove_switch()
    update_graph(no_switch_gd)

    fold_bn_gd = fold_bn()
    update_graph(fold_bn_gd)

    # opt end #

    with tf.Session() as sess:
        final_gd = tf.graph_util.convert_variables_to_constants(
            sess, tf.get_default_graph().as_graph_def(), ['final_dense/MatMul'])

    ensure_graph_is_valid(final_gd)

    tf.summary.FileWriter(OUTPUT_PATH, graph=final_gd)

    tf.io.write_graph(final_gd, "../pb/", "frozen_shape_28.pb", as_text=False)
    tf.io.write_graph(final_gd, "../pb/", "frozen_shape_28.pbtxt", as_text=True)


def update_graph(graph_def):
    tf.reset_default_graph()
    tf.import_graph_def(graph_def, name='')


def test():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile('../pb/frozen_shape_28.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        input_node = tf.get_default_graph().get_tensor_by_name(
            "new_input_node:0")
        output_node = tf.get_default_graph().get_tensor_by_name(
            "final_dense/MatMul:0")

        image = cv2.imread('../05/ithome.jpg')
        image = cv2.resize(image, (128, 128))

        output = sess.run(output_node, feed_dict={input_node: np.expand_dims(image, 0)})
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
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        if new_node.name == target_node_name:
            del new_node.input[:]
            for input_name in input_before_removal:
                if input_name == old_input_name:  # replace old input
                    new_node.input.append(mul.op.name)
                else:
                    new_node.input.append(input_name)  # keep old input
        nodes_after_modify.append(new_node)

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_modify)
    return new_gd


def remove_training_node(is_training_node, false_node_name):
    false_node = tf.constant(False, dtype=tf.bool, shape=(), name=false_node_name)

    old_gd = tf.get_default_graph().as_graph_def()
    old_nodes = old_gd.node  # old nodes from graph

    nodes_after_modify = []
    for node in old_nodes:
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:
            if full_input_name == is_training_node:
                new_node.input.append(false_node.op.name)
            else:
                new_node.input.append(full_input_name)
        nodes_after_modify.append(new_node)

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_modify)
    return new_gd


def remove_dropout():
    old_gd = tf.get_default_graph().as_graph_def()
    new_gd = strip_dropout(old_gd,
                           drop_scope='backend/dropout_2',
                           dropout_before='backend/relu_2',
                           dropout_after='backend/max_pool_2/MaxPool')

    return new_gd


def strip_dropout(input_graph, drop_scope, dropout_before, dropout_after):
    input_nodes = input_graph.node
    nodes_after_strip = []
    for node in input_nodes:  # for所有節點
        if node.name.startswith(drop_scope + '/'):  # drop_scope底下跳過
            continue

        new_node = node_def_pb2.NodeDef()  # 產生新節點
        new_node.CopyFrom(node)  # 拷貝舊節點資訊到新節點
        if new_node.name == dropout_after:  # 若該節點是after
            new_input = []
            for input_name in new_node.input:  # 檢查該節點的input
                if input_name.startswith(drop_scope + '/'):
                    new_input.append(dropout_before)  # 若是drop_scope改成前節點
                else:
                    new_input.append(input_name)  # 若不是，保持原input
            del new_node.input[:]
            new_node.input.extend(new_input)  # 更新input
        nodes_after_strip.append(new_node)  # 將新節點存到list

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_strip)
    return new_gd


def remove_merge():
    old_gd = tf.get_default_graph().as_graph_def()
    old_nodes = old_gd.node

    names_to_remove = {}  # 產生一個dict記錄等等哪些節點要被砍掉
    for node in old_nodes:  # for所有節點
        if node.op == 'Merge':  # 當發現節點是Merge時
            # FusedBatchNorm_1=0, FusedBatchNorm=1
            names_to_remove[node.name] = node.input[0]  # 將節點存到key，位置0存到value
            names_to_remove[node.input[1]] = None  # 位置1因為要被砍，所以只存key
    print(f'remove merge: {names_to_remove}')

    nodes_after_modify = []
    for node in old_nodes:  # for所有節點
        if node.name in names_to_remove:
            continue  # 如果是要被砍掉的就跳過
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        input_before_removal = node.input
        del new_node.input[:]
        for full_input_name in input_before_removal:  # 檢查input節點
            input_name = re.sub(r"^\^", "", full_input_name)  # 去除名稱中其他符號
            while input_name in names_to_remove:  # 如果input_name是要被砍的，就進迴圈
                full_input_name = names_to_remove[input_name]  # 找上個輸入
                input_name = re.sub(r"^\^", "", full_input_name)  # 去除名稱中其他符號
            new_node.input.append(full_input_name)  # 把FusedBatchNorm_1加到新節點
        nodes_after_modify.append(new_node)  # 新增該節點

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_modify)
    return new_gd


def remove_switch():
    old_gd = tf.get_default_graph().as_graph_def()
    old_nodes = old_gd.node

    names_to_remove = {}  # 產生一個dict記錄哪些節點要被砍掉
    for node in old_nodes:  # for所有節點
        if node.op != 'Switch':  # 當發現節點不是Switch時跳過
            continue
        for node_i in node.input:  # 檢查Switch的input
            # 若input是pred_id,把pred_id以外的input存到dict
            if node_i.split('/')[-1] == 'pred_id':
                names_to_remove[node.name] = [
                    x for x in node.input if x.split('/')[-1] != 'pred_id'
                ]
    print(f'remove switch: {names_to_remove}')

    nodes_after_modify = []
    for node in old_nodes:  # for所有節點
        if node.name in names_to_remove:
            continue  # 如果是要被砍掉的就跳過
        new_node = node_def_pb2.NodeDef()  # 產生新節點
        new_node.CopyFrom(node)  # 拷貝舊節點資訊
        input_before_removal = node.input
        del new_node.input[:]  # 去掉所有input
        for full_input_name in input_before_removal:
            if full_input_name in names_to_remove:  # 如果此節點input是switch
                for input_name in names_to_remove[full_input_name]:
                    # 就把names_to_remove的所有input補上
                    new_node.input.append(input_name)
            else:  # 不然一切照舊
                new_node.input.append(full_input_name)
        nodes_after_modify.append(new_node)

    new_gd = graph_pb2.GraphDef()
    new_gd.node.extend(nodes_after_modify)
    return new_gd


def fold_bn():
    old_gd = tf.get_default_graph().as_graph_def()
    new_gd = fold(old_gd,
                  bn_scope='backend/bn_1',
                  conv_scope='backend/conv_1')
    new_gd = fold(new_gd,
                  bn_scope='backend/bn_2',
                  conv_scope='backend/conv_2')
    new_gd = fold(new_gd,
                  bn_scope='backend/bn_3',
                  conv_scope='backend/conv_3')
    return new_gd


def fold(graph_def, bn_scope, conv_scope):
    nodes = graph_def.node

    #  get values and nodes  #
    mean_node, mean_value = values_from_const(nodes, f'{bn_scope}/moving_mean')
    var_node, var_value = values_from_const(nodes, f'{bn_scope}/moving_variance')
    epison_value = values_from_attr(nodes, f'{bn_scope}/cond/FusedBatchNorm_1', 'epsilon')

    gamma_node, gamma_value = values_from_const(nodes, f'{bn_scope}/gamma')
    beta_node, beta_value = values_from_const(nodes, f'{bn_scope}/beta')

    kernel_node, kernel_value = values_from_const(nodes, f'{conv_scope}/kernel')
    bias_node, bias_value = values_from_const(nodes, f'{conv_scope}/bias')

    fused_bn_node = node_from_name(nodes, f'{bn_scope}/cond/FusedBatchNorm_1')
    conv_node = node_from_name(nodes, f'{conv_scope}/Conv2D')
    bias_add_node = node_from_name(nodes, f'{conv_scope}/BiasAdd')

    # 以下四個節點會拿去運算,新圖表不會產生
    nodes_to_skip = [kernel_node.name, bias_node.name, fused_bn_node.name, bias_add_node.name]

    #  fold bn values  #
    scale_value = ((1.0 / np.vectorize(
        math.sqrt)(var_value + epison_value)) * gamma_value)

    scaled_weights = np.copy(kernel_value)
    it = np.nditer(scaled_weights, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        current_scale = scale_value[it.multi_index[3]]
        it[0] *= current_scale
        it.iternext()
    offset_value = (bias_value - mean_value) * scale_value + beta_value

    # 新的conv weight 常數節點
    scaled_weights_op = node_def_pb2.NodeDef()
    scaled_weights_op.op = "Const"
    scaled_weights_op.name = kernel_node.name
    scaled_weights_op.attr["dtype"].CopyFrom(kernel_node.attr["dtype"])
    scaled_weights_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                scaled_weights, kernel_value.dtype.type, kernel_value.shape)))
    # 新的conv bias 常數節點
    offset_op = node_def_pb2.NodeDef()
    offset_op.op = "Const"
    offset_op.name = bias_node.name
    offset_op.attr["dtype"].CopyFrom(bias_node.attr["dtype"])
    offset_op.attr["value"].CopyFrom(
        attr_value_pb2.AttrValue(
            tensor=tensor_util.make_tensor_proto(
                offset_value, bias_value.dtype.type, bias_value.shape)))
    # 新的conv weight bias 合併運算節點
    bias_add_op = node_def_pb2.NodeDef()
    bias_add_op.op = "BiasAdd"
    # 雖然是BiasAdd,但名稱仍叫FusedBatchNorm_1
    bias_add_op.name = fused_bn_node.name
    bias_add_op.attr["T"].CopyFrom(bias_add_node.attr["T"])
    bias_add_op.attr["data_format"].CopyFrom(bias_add_node.attr["data_format"])
    bias_add_op.input.extend([conv_node.name, offset_op.name])

    #  gen graph def  #
    new_gd = graph_pb2.GraphDef()
    for node in nodes:  # for所有節點
        if node.name in nodes_to_skip:
            # 該skip的就skip
            continue
        new_node = node_def_pb2.NodeDef()
        new_node.CopyFrom(node)
        new_gd.node.extend([new_node])

    # 除了原本該有的節點外多新增以下三個節點
    new_gd.node.extend([scaled_weights_op, offset_op, bias_add_op])
    return new_gd


def node_from_name(nodes, node_name):
    for node in nodes:
        if node.name == node_name:
            return node
    raise Exception(f'Can\'t find {node_name}')


def values_from_const(nodes, node_name):
    for node in nodes:
        if node.name == node_name:
            input_tensor = node.attr['value'].tensor
            value = tensor_util.MakeNdarray(input_tensor)
            return node, value
    raise Exception(f'Can\'t find {node_name}')


def values_from_attr(nodes, node_name, key):
    for node in nodes:
        if node.name == node_name:
            value = node.attr[key].f
            return value
    raise Exception(f'Can\'t find {node_name}')


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
