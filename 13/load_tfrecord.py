import tensorflow as tf

TFRECORD_PATH = '../tfrecord/member.tfrecord'


def main():
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORD_PATH)

    for tf_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(tf_record)

        name = example.features.feature['member/name'].bytes_list.value[
            0].decode('utf8')
        encoded_image = example.features.feature[
            'member/encoded'].bytes_list.value[0]
        with tf.gfile.GFile(f'img/{name}.png', 'wb') as fid:
            fid.write(encoded_image)

        age = example.features.feature['member/age'].int64_list.value[0]
        height = example.features.feature['member/height'].float_list.value[0]
        prefer_prods = example.features.feature[
            'member/prefer_prods'].int64_list.value

        print(f'user: {name}')
        print(f'age: {age}')
        print(f'height: {height}')
        print(f'prefer_prdos: {prefer_prods}\n')


if __name__ == '__main__':
    main()
