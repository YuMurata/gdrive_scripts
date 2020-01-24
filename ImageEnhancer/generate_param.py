from .enhance_definer import enhance_name_list, MAX_PARAM, MIN_PARAM
from random import random


def generate_random_param() -> dict:
    return dict(zip(enhance_name_list,
                    [(MAX_PARAM - MIN_PARAM)*random()+MIN_PARAM
                     for _ in range(len(enhance_name_list))]))


def generate_random_param_list(generate_num: int) -> list:
    return [generate_random_param() for _ in range(generate_num)]
