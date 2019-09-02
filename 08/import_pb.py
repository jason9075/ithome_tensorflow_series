import tensorflow as tf
from google.protobuf import text_format
from tensorflow.python.platform import gfile

OUTPUT_PATH = '../events/'

MODEL_PB = '../pb/model.pb'
MODEL_PBTXT = '../pb/model.pbtxt'
FROZEN_PB = '../pb/frozen_model.pb'
FROZEN_PBTXT = '../pb/frozen_model.pbtxt'


def read_pb():
    graph_def = tf.get_default_graph().as_graph_def()
    with gfile.FastGFile(MODEL_PB, 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


def read_pb_txt():
    # this will cause error. "should not have multiple "versions" fields."
    # graph_def = tf.get_default_graph().as_graph_def()
    graph_def = tf.GraphDef()
    with gfile.FastGFile(MODEL_PBTXT, 'rb') as f:
        text_format.Parse(f.read(), graph_def)
    tf.import_graph_def(graph_def, name='')

    tf.summary.FileWriter(OUTPUT_PATH, graph=tf.get_default_graph())


if __name__ == '__main__':
    # read_pb()
    read_pb_txt()
