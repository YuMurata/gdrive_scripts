import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

import json
import config
from make_traindata import ParamDistance
from submodule import ImageRankNet, TrainDataMaker
from argparse import ArgumentParser
from ImageEnhancer import generate_random_param, ResizableEnhancer
from tqdm import tqdm


def _get_args():
    parser = ArgumentParser()

    parser.add_argument('-u', '--user_name', required=True)
    parser.add_argument('-i', '--image_category_name', required=True)
    parser.add_argument('-n', '--iteration_num', type=int, default=1000)

    args = parser.parse_args()

    for arg in vars(args):
        print(f'{str(arg)}: {str(getattr(args, arg))}')

    return args


if __name__ == "__main__":
    args = _get_args()

    ranknet = ImageRankNet.RankNet(config.ImageInfo.shape)
    weight_path = str(config.DirectoryPath.weight /
                      args.user_name/f'{args.image_category_name}.h5')

    ranknet.load(weight_path)

    scored_param_path = str(
        config.DirectoryPath.scored_param/args.user_name/f'{args.image_category_name}.json')
    with open(scored_param_path, 'r') as fp:
        scored_param = json.load(fp)

    scored_player_list = [TrainDataMaker.Player(x['param'], x['score'])
                          for x in scored_param]

    evaluator = TrainDataMaker.Evaluator(scored_player_list, ParamDistance())

    enhancer = ResizableEnhancer(
        config.ImagePath.image_path_dict[args.image_category_name],
        config.ImageInfo.size)

    miss_predict_num = 0

    for _ in tqdm(range(args.iteration_num), desc='evaluate'):
        left_param = generate_random_param()
        right_param = generate_random_param()

        left_image = enhancer.resized_enhance(left_param)
        right_image = enhancer.resized_enhance(right_param)

        left_score = evaluator.evaluate(left_param)
        right_score = evaluator.evaluate(right_param)

        predict = ranknet.predict([left_image, right_image])
        left_predict = predict[0][0]
        right_predict = predict[1][0]

        if left_score > right_score:
            if not left_predict > right_predict:
                miss_predict_num += 1

        if left_score < right_score:
            if not left_predict < right_predict:
                miss_predict_num += 1

        if left_score == right_score:
            if not left_predict == right_predict:
                miss_predict_num += 1

    print(f'error: {miss_predict_num/args.iteration_num}')
