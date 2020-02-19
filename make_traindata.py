from argparse import ArgumentParser
from submodule import TrainDataMaker
from ImageEnhancer \
    import (enhance_name_list, ResizableEnhancer, generate_random_param)
import json
import config
import numpy as np


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-n', '--generate_num', required=True, type=int)
    parser.add_argument('-i', '--image_name', required=True)
    parser.add_argument('-u', '--user_name', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


class ParamDistance(TrainDataMaker.DistanceMeasurer):
    def measure(self, paramA: dict, paramB: dict):
        return sum([abs(paramA[enhance_name] - paramB[enhance_name])
                    for enhance_name in enhance_name_list])


class EnhanceGenerator(TrainDataMaker.DataGenerator):
    def __init__(self, image_path: str):
        self.enhancer = ResizableEnhancer(image_path, config.ImageInfo.size)

    def generate(self, param):
        return np.array(self.enhancer.resized_enhance(param))


class EnhanceParamGenerator(TrainDataMaker.RandomParamGenerator):
    def generate(self):
        return generate_random_param()


if __name__ == "__main__":
    args = _get_args()

    try:
        scored_param_dir_path = \
            config.DirectoryPath.scored_param / \
            args.user_name / args.image_name

        tfrecords_dir_path = config.DirectoryPath.tfrecords / \
            args.user_name / args.image_name
        tfrecords_dir_path.mkdir(parents=True, exist_ok=True)

        train_path = tfrecords_dir_path / 'train.tfrecords'
        is_train_exists = train_path.exists()
        train_writer = TrainDataMaker.Writer(str(train_path))
        if is_train_exists:
            print(f'{str(train_path)} is already exist')

        valid_path = tfrecords_dir_path / 'validation.tfrecords'
        is_valid_exists = valid_path.exists()
        valid_writer = TrainDataMaker.Writer(str(valid_path))
        if is_valid_exists:
            print(f'{str(valid_path)} is already exist')

        scored_param_num = len(list(scored_param_dir_path.iterdir()))
        generate_per_param = args.generate_num//scored_param_num

        for scored_param_path in scored_param_dir_path.iterdir():
            with open(str(scored_param_path), 'r') as fp:
                scored_param = json.load(fp)

            scored_player_list = [TrainDataMaker.Player(x['param'], x['score'])
                                  for x in scored_param]

            evaluator = TrainDataMaker.Evaluator(
                scored_player_list, ParamDistance())

            image_dir_path = config.DirectoryPath.image / args.image_name
            if not image_dir_path.exists():
                raise FileNotFoundError('invalid image name')

            image_path_candidate = \
                [image_path for image_path in image_dir_path.iterdir()
                 if image_path.stem == scored_param_path.stem]
            if len(image_path_candidate) != 1:
                raise Exception(f'{scored_param_path.stem} is multiple exists')
            image_path = str(image_path_candidate[0])
            enhancer = EnhanceGenerator(image_path)

            if not is_train_exists:
                TrainDataMaker.make_tfrecords(
                    train_writer,
                    generate_per_param,
                    EnhanceParamGenerator(),
                    enhancer, evaluator)

            if not is_valid_exists:
                TrainDataMaker.make_tfrecords(
                    valid_writer,
                    generate_per_param // 10, EnhanceParamGenerator(),
                    enhancer, evaluator)

    except FileNotFoundError as e:
        print(e)
