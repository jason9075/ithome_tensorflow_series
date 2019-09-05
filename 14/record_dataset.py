import cv2
import tensorflow as tf

TFRECORD_PATH = '../tfrecord/member.tfrecord'


def main():
    data_set = tf.data.TFRecordDataset(TFRECORD_PATH)
    data_set = data_set.map(parse_function)
    data_set = data_set.shuffle(buffer_size=9)
    data_set = data_set.batch(3)
    iterator = data_set.make_initializable_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
        sess.run(iterator.initializer)

        results, imgs = sess.run(next_element)

        print('names: {}'.format(results['member/name']))
        print('ages: {}'.format(results['member/age']))
        print('heights: {}'.format(results['member/height']))
        print('prefer_prods: {}'.format(results['member/prefer_prods']))

        for img in imgs:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow('img', img)
            cv2.waitKey(-1)


def parse_function(example_proto):
    features = {'member/name': tf.io.FixedLenFeature([], tf.string),
                'member/encoded': tf.io.FixedLenFeature([], tf.string),
                'member/age': tf.io.FixedLenFeature([], tf.int64),
                'member/height': tf.io.VarLenFeature(tf.float32),
                'member/prefer_prods': tf.io.VarLenFeature(tf.int64)}
    features = tf.io.parse_single_example(example_proto, features)
    images = tf.image.decode_png(features['member/encoded'], channels=3)
    # 注意png原本有4個channel，但執行到下面的處理會出錯，所以前一行先降成3個channel。
    images = tf.image.random_brightness(images, 0.1)
    images = tf.image.random_saturation(images, 0.7, 1.3)
    images = tf.image.random_contrast(images, 0.6, 1.5)
    images = tf.image.random_flip_left_right(images)

    return features, images


if __name__ == '__main__':
    main()
