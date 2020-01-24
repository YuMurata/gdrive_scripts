import tensorflow as tf
from pathlib import Path
from .exception import ModelException

SCOPE = 'ranknet_dataset'


class DatasetException(ModelException):
    pass


def make_dataset(dataset_file_path: str, batch_size: int, name: str, image_shape: tuple) -> tf.data.TFRecordDataset:
    dataset_file_path = Path(dataset_file_path)
    if not dataset_file_path.exists():
        raise DatasetException(f'{str(dataset_file_path)} is not found')

    def _parse_function(example_proto):
        features = {
            'label': tf.io.FixedLenFeature((), tf.int64,
                                           default_value=0),
            'left_image': tf.io.FixedLenFeature((), tf.string,
                                                default_value=""),
            'right_image': tf.io.FixedLenFeature((), tf.string,
                                                 default_value=""),
        }
        parsed_features = tf.io.parse_single_example(example_proto, features)

        return parsed_features

    def _read_image(parsed_features):
        left_image_raw = \
            tf.io.decode_raw(parsed_features['left_image'], tf.uint8)
        right_image_raw =\
            tf.io.decode_raw(parsed_features['right_image'], tf.uint8)

        label = tf.cast(parsed_features['label'], tf.int32, name='label')

        float_left_image_raw = tf.cast(left_image_raw, tf.float32)/255
        float_right_image_raw = tf.cast(right_image_raw, tf.float32)/255

        def _augmentation(image):
            width, height, channel = image_shape

            x = tf.image.random_flip_left_right(image)
            x = tf.image.random_crop(
                x, [int(width*0.8), int(height*0.8), channel])
            x = tf.image.resize(x, (width, height))
            return x

        left_image = \
            tf.reshape(float_left_image_raw, image_shape, name='left_image')
        left_image = _augmentation(left_image)

        right_image = \
            tf.reshape(float_right_image_raw, image_shape, name='right_image')
        right_image = _augmentation(right_image)

        return ((left_image, right_image), label)

    with tf.name_scope(f'{name}_{SCOPE}'):
        dataset = \
            tf.data.TFRecordDataset(str(dataset_file_path)) \
            .map(_parse_function) \
            .map(_read_image) \
            .shuffle(batch_size) \
            .batch(batch_size) \
            .repeat()

    return dataset
