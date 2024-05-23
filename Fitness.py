
class Fitness:
    def __init__(self, params={}, method='high_score'):
        self.params = params
        self.method = method
        self.methods = {
            'high_score': self.high_score,
            'score_death': self.score_death
        }

    def __repr__(self) -> str:
        return self.method

    def __call__(self, result):
        if self.method in self.methods:
            return self.methods[self.method](result)
        else:
            raise ValueError(f"Unknown method: {self.method}")


    def high_score(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, same_dir_as_before, moves = result.values()
        score_weight = 900000
        penalty=0
        score = score*params['score']
        high_score = high_score*params['high_score']
        death = -1*(death*params['death'])
        moves_with_out_food = -1*(moves_with_out_food*params['moves_without_food'])
        death_no_food = -1*(death_no_food*params['death_no_food'])
        fit = high_score+death+moves_with_out_food+death_no_food+penalty

        return fit

    def score_death(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, same_dir_as_before, moves = result.values()

        score = score*params['score']
        same_dir_as_before = params['same_dir_as_befire']*(same_dir_as_before)
        death = -1*(death*params['death'])
        if death > 0:
            -1*(death/score+1)
        fit = 0
        fit=fit+score+death
        return fit
