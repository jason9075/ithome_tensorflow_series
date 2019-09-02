import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

MODEL_PB = '../pb/bn_dropout_model.pb'


def main():
    graph = tf.get_default_graph()

    graph_def = graph.as_graph_def()
    with gfile.FastGFile(MODEL_PB, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    input_node = graph.get_tensor_by_name(
        "input_node:0")
    training_node = graph.get_tensor_by_name(
        "training:0")

    debug_node = graph.get_tensor_by_name(
        "bn/cond/Merge:0")

    with tf.Session() as sess:
        image = cv2.imread('../05/ithome.jpg')
        image = np.expand_dims(image, 0)

        result = sess.run(debug_node, feed_dict={input_node: image, training_node: True})
        print(f'training true:\n{result[0, 22:28, 22:28, 0]}')

        result = sess.run(debug_node, feed_dict={input_node: image, training_node: False})
        print(f'training false:\n{result[0, 22:28, 22:28, 0]}')


if __name__ == '__main__':
    main()
