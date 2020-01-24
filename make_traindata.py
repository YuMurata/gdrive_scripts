from argparse import ArgumentParser
from TrainDataMaker import Evaluator, Player, make_tfrecords
from ImageEnhancer import enhance_name_list, ResizableEnhancer
import json
import config
from pathlib import Path


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


def _param_diff(param, target_param):
    return -sum([abs(param[enhance_name]-target_param[enhance_name]) for enhance_name in enhance_name_list])


if __name__ == "__main__":
    args = _get_args()

    try:
        with open(args.scored_json_path, 'r') as fp:
            scored_json = json.load(fp)

        scored_player_list = [Player(x['param'], x['score'])
                              for x in scored_json]

        evaluator = Evaluator(scored_player_list, _param_diff)

        if args.image_name not in config.image_path_dict:
            raise FileNotFoundError('invalid image name')

        image_path = config.image_path_dict[args.image_name]
        enhancer = ResizableEnhancer(image_path, config.IMAGE_SIZE)

        save_dir_path = Path(args.save_dir_path)
        save_dir_path.mkdir(parents=True, exist_ok=True)

        make_tfrecords(str(save_dir_path/'train.tfrecords'),
                       args.generate_num, enhancer, evaluator)

        make_tfrecords(str(save_dir_path/'validation.tfrecords'),
                       args.generate_num//10, enhancer, evaluator)
    except FileNotFoundError as e:
        print(e)
