
class Fitness:
    def __init__(self, params={}, method='high_score'):
        self.params = params
        self.method = method
        self.methods = {
            'high_score': self.high_score,
            'step_death': self.step_death
        }

    def __call__(self, result):
        if self.method in self.methods:
            return self.methods[self.method](result)
        else:
            raise ValueError(f"Unknown method: {self.method}")


    def high_score(self, result):
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, moves = result.values()
        score_weight = 900000
        penalty=0
        score = score*1000
        high_score = high_score*100
        death = -1*(death*100)
        moves_with_out_food = -1*(moves_with_out_food*30)
        death_no_food = -1*(death_no_food*100)
        fit = high_score+death+moves_with_out_food+death_no_food+penalty

        return fit

    def step_death(self, result):
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, moves = result.values()

        score = score*1000
        high_score = high_score*100
        death = -1*(death*100)
        moves_with_out_food = -1*(moves_with_out_food*30)
        death_no_food = -1*(death_no_food*100)
        fit = high_score+death+moves_with_out_food+death_no_food

        return fit
