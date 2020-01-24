import typing
from .player import Player


class Evaluator:
    def __init__(self, scored_player_list: typing.List[Player], param_diff_func: typing.Callable[[typing.Any, typing.Any], float]):
        self.scored_player_list = scored_player_list
        self.sum_score = sum(
            [player.score for player in self.scored_player_list])
        self.param_diff_func = param_diff_func

    def evaluate(self, param):
        return sum([player.score*self.param_diff_func(param, player.param) for player in self.scored_player_list])/self.sum_score
