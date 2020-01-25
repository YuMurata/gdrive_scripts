from submodule import ImageRankNet
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import config
import tensorflow as tf

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

DATASET_TYPE_LIST = [TRAIN, VALIDATION, TEST]
SUFFIX = '.tfrecords'


def _make_summary_dir(summary_dir_path: str):
    now = datetime.now()
    path = Path(summary_dir_path) / \
        '{0:%m%d}'.format(now) / '{0:%H%M}'.format(now)

    if path.exists():
        path = Path(str(path.parent) + '_{0:%S}'.format(now))

    path.mkdir(parents=True)

    return str(path)


def _make_dataset_path_dict(dataset_dir_path: str):
    dataset_dir_path = Path(dataset_dir_path)

    if not dataset_dir_path.exists():
        raise FileNotFoundError('フォルダが見つかりませんでした')
    elif not dataset_dir_path.is_dir():
        raise NotADirectoryError

    dataset_path_dict = \
        {key: str(dataset_dir_path / (key + SUFFIX))
         for key in DATASET_TYPE_LIST}
    return dataset_path_dict


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-u', '--user_name', required=True)
    parser.add_argument('-i', '--image_name', required=True)

    parser.add_argument('-l', '--is_load', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


class ImageMapper(ImageRankNet.dataset.Mapper):
    def map_example(self, example_proto):
        features = {
            'label': tf.io.FixedLenFeature((), tf.int64,
                                           default_value=0),
            'left_image': tf.io.FixedLenFeature((), tf.string,
                                                default_value=""),
            'right_image': tf.io.FixedLenFeature((), tf.string,
                                                 default_value=""),
        }

        parsed_features = tf.io.parse_single_example(example_proto, features)

        left_image_raw = \
            tf.io.decode_raw(parsed_features['left_image'], tf.uint8)
        right_image_raw =\
            tf.io.decode_raw(parsed_features['right_image'], tf.uint8)

        label = tf.cast(parsed_features['label'], tf.int32, name='label')

        float_left_image_raw = tf.cast(left_image_raw, tf.float32) / 255
        float_right_image_raw = tf.cast(right_image_raw, tf.float32) / 255

        image_shape = config.ImageInfo.shape

        def _augmentation(image):
            width, height, channel = image_shape

            x = tf.image.random_flip_left_right(image)
            x = tf.image.random_crop(
                x, [int(width * 0.8), int(height * 0.8), channel])
            x = tf.image.resize(x, (width, height))
            return x

        left_image = \
            tf.reshape(float_left_image_raw, image_shape, name='left_image')
        left_image = _augmentation(left_image)

        right_image = \
            tf.reshape(float_right_image_raw, image_shape, name='right_image')
        right_image = _augmentation(right_image)

        return ((left_image, right_image), label)


class MyCNN(ImageRankNet.EvaluateBody):
    def __init__(self):
        super().__init__(config.ImageInfo.shape)

    def build(self):
        model = tf.keras.Sequential()

        model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=5,
                                         padding='same', activation='relu',
                                         input_shape=self.image_shape))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=5,
                                         padding='same', activation='relu'))
        model.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

        return model


if __name__ == "__main__":
    args = _get_args()

    trainable_model = ImageRankNet.RankNet(MyCNN())

    load_file_path = \
        config.DirectoryPath.weight / args.user_name / f'{args.image_name}.h5'
    if args.is_load and load_file_path.exists() and load_file_path.is_file():
        trainable_model.load(str(load_file_path))

    dataset_path_dict = _make_dataset_path_dict(
        str(config.DirectoryPath.tfrecords / args.user_name / args.image_name))

    dataset = {key: ImageRankNet.dataset.make_dataset(dataset_path_dict[key],
                                                      ImageMapper(),
                                                      args.batch_size, key
                                                      )
               for key in [TRAIN, VALIDATION]}

    weight_dir_path = config.DirectoryPath.weight / args.user_name
    weight_dir_path.mkdir(exist_ok=True, parents=True)

    log_dir_path = weight_dir_path / 'logs'
    log_dir_path.mkdir(exist_ok=True, parents=True)

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            str(weight_dir_path / f'{args.image_name}.h5'),
            monitor='val_loss',
            verbose=0, save_best_only=True,
            save_weights_only=True, mode='auto',
            period=1),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(log_dir_path), write_graph=True)
    ]

    trainable_model.train(dataset[TRAIN], dataset[VALIDATION],
                          callback_list=callback_list, epochs=args.epochs,
                          steps_per_epoch=30)
