import timeit

import cv2
import numpy as np
import tensorflow as tf

TIMES = 1000


def main():

    interpreter = tf.lite.Interpreter(model_path="../tflite/model.lite")
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    image = cv2.imread('../05/ithome.jpg')
    image = cv2.resize(image, (128, 128))
    image = image.astype(np.float32)

    start = timeit.default_timer()
    for _ in range(0, TIMES):
        interpreter.set_tensor(input_index, np.expand_dims(image, axis=0))
        interpreter.invoke()
        interpreter.get_tensor(output_index)

    print(f'cost time:{(timeit.default_timer() - start)} sec')


if __name__ == '__main__':
    main()
