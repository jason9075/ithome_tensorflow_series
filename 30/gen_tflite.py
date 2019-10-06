import tensorflow as tf


def main():
    converter = tf.lite.TFLiteConverter.from_frozen_graph('../pb/frozen_shape_28.pb',
                                                          ['new_input_node'], ['final_dense/MatMul'])
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
    tflite_model = converter.convert()

    with open("../tflite/model.lite", "wb") as f:
        f.write(tflite_model)


if __name__ == '__main__':
    main()
