from argparse import ArgumentParser
from submodule \
    import (Evaluator, Player, make_tfrecords, DistanceMeasurer,
            DataGenerator, RandomParamGenerator)
from ImageEnhancer \
    import (enhance_name_list, ResizableEnhancer, generate_random_param)
import json
import config
from pathlib import Path
import numpy as np


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-j', '--scored_json_path', required=True)
    parser.add_argument('-s', '--save_dir_path', required=True)
    parser.add_argument('-n', '--generate_num', required=True, type=int)
    parser.add_argument('-i', '--image_name', required=True)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


class ParamDistance(DistanceMeasurer):
    def measure(paramA: dict, paramB: dict):
        return -sum([abs(paramA[enhance_name]-paramB[enhance_name])
                     for enhance_name in enhance_name_list])


class EnhanceGenerator(DataGenerator):
    def __init__(self, image_path: str):
        self.enhancer = ResizableEnhancer(image_path, config.IMAGE_SIZE)

    def generate(self, param):
        return np.array(self.enhancer.resized_enhance(param))


class EnhanceParamGenerator(RandomParamGenerator):
    def generate(self):
        return generate_random_param()


if __name__ == "__main__":
    args = _get_args()

    try:
        with open(args.scored_json_path, 'r') as fp:
            scored_json = json.load(fp)

        scored_player_list = [Player(x['param'], x['score'])
                              for x in scored_json]

        evaluator = Evaluator(scored_player_list, ParamDistance())

        if args.image_name not in config.image_path_dict:
            raise FileNotFoundError('invalid image name')

        image_path = config.image_path_dict[args.image_name]
        enhancer = EnhanceGenerator(image_path)

        save_dir_path = Path(args.save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        make_tfrecords(str(save_dir_path/'train.tfrecords'),
                       args.generate_num, enhancer, evaluator)

        make_tfrecords(str(save_dir_path/'validation.tfrecords'),
                       args.generate_num//10, enhancer, evaluator)
    except FileNotFoundError as e:
        print(e)
