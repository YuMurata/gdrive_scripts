class Player:
    def __init__(self, param, score: int = 1):
        self.param = param
        self.score = score

    def score_up(self):
        self.score *= 2

    def to_dict(self):
        return {'score': self.score, 'param': self.param}
