import tensorflow as tf

TFRECORD_PATH = '../tfrecord/member.tfrecord'


def main():
    record_iterator = tf.python_io.tf_record_iterator(path=TFRECORD_PATH)

    for tf_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(tf_record)

        name = get_bytes(example, 'member/name').decode('utf8')
        encoded_image = get_bytes(example, 'member/encoded')
        with tf.gfile.GFile(f'img/{name}.png', 'wb') as fid:
            fid.write(encoded_image)

        age = get_int64(example, 'member/age')
        height = get_float(example, 'member/height')
        prefer_prods = get_int64_list(example, 'member/prefer_prods')

        print(f'user: {name}')
        print(f'age: {age}')
        print(f'height: {height:.1f}')
        print(f'prefer_prdos: {prefer_prods}\n')


def get_int64(example, key):
    return example.features.feature[key].int64_list.value[0]


def get_int64_list(example, key):
    return example.features.feature[key].int64_list.value


def get_bytes(example, key):
    return example.features.feature[key].bytes_list.value[0]


def get_float(example, key):
    return example.features.feature[key].float_list.value[0]


if __name__ == '__main__':
    main()
