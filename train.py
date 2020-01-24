import Model
from argparse import ArgumentParser

from datetime import datetime
from pathlib import Path

TRAIN = 'train'
VALIDATION = 'validation'
TEST = 'test'

DATASET_TYPE_LIST = [TRAIN, VALIDATION, TEST]
SUFFIX = '.tfrecords'


class TrainModelException(Model.ModelException):
    pass


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
    parser.add_argument('-s', '--summary_dir_path', required=True)
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
    def get_summary_dir_path_func():
        return _make_summary_dir(args.summary_dir_path)

    args = _get_args()

    trainable_model = Model.RankNet(args.image_shape, use_vgg16=args.use_vgg16)
    if args.load_file_path is not None:
        load_file_path = Path(args.load_file_path)
        if load_file_path.exists() and load_file_path.is_file():
            trainable_model.load(args.load_file_path)

    dataset_path_dict = _make_dataset_path_dict(args.dataset_dir_path)

    try:
        dataset = {key: Model.make_dataset(dataset_path_dict[key], args.batch_size, key, args.image_shape)
                   for key in [TRAIN, VALIDATION]}
    except Model.DatasetException as e:
        raise TrainModelException(e)

    summary_dir_path = get_summary_dir_path_func()
    trainable_model.train(dataset[TRAIN], log_dir_path=summary_dir_path,
                          valid_dataset=dataset[VALIDATION], epochs=args.epochs, steps_per_epoch=30)
