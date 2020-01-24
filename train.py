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
    path = Path(summary_dir_path)/'{0:%m%d}'.format(now)/'{0:%H%M}'.format(now)

    if path.exists():
        path = Path(str(path.parent)+'_{0:%S}'.format(now))

    path.mkdir(parents=True)

    return str(path)


def _make_dataset_path_dict(dataset_dir_path: str):
    dataset_dir_path = Path(dataset_dir_path)

    if not dataset_dir_path.exists():
        raise FileNotFoundError('フォルダが見つかりませんでした')
    elif not dataset_dir_path.is_dir():
        raise NotADirectoryError

    dataset_path_dict = \
        {key: str(dataset_dir_path/(key+SUFFIX))
         for key in DATASET_TYPE_LIST}
    return dataset_path_dict


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-d', '--dataset_dir_path', required=True)
    parser.add_argument('-u', '--user_name', required=True)
    parser.add_argument('-i', '--image_name', required=True)
    parser.add_argument('-sh', '--image_shape', nargs=3,
                        required=True, type=int)
    parser.add_argument('-l', '--load_file_path')
    parser.add_argument('-vgg', '--use_vgg16', action='store_true')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=100)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    trainable_model = ImageRankNet.RankNet(args.image_shape,
                                           use_vgg16=args.use_vgg16)

    load_file_path = config.DirectoryPath.weight/args.user_name/f'{args.image_name}.h5'
    if load_file_path.exists() and load_file_path.is_file():
        trainable_model.load(args.load_file_path)

    dataset_path_dict = _make_dataset_path_dict(args.dataset_dir_path)

    dataset = {key: ImageRankNet.dataset.make_dataset(dataset_path_dict[key],
                                                      args.batch_size, key,
                                                      args.image_shape)
               for key in [TRAIN, VALIDATION]}

    weight_dir_path = config.DirectoryPath.weight/args.user_name
    weight_dir_path.mkdir(exist_ok=True, parents=True)

    log_dir_path = weight_dir_path/'logs'
    log_dir_path.mkdir(exist_ok=True, parents=True)

    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(str(weight_dir_path/'weight.h5'),
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
