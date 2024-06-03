
class Fitness:
    def __init__(self, params={}, method='high_score'):
        self.params = params
        self.method = method
        self.methods = {
            'high_score': self.high_score,
            'score_death': self.score_death,
            'move_no_food': self.move_no_food,
            'train_death': self.train_death,
            'kejitech':self.kejitech,
            'food_death':self.food_death,
            'food_death_simple':self.food_death_simple,
            'death_simple':self.death_simple,
            "food_step_single_life":self.food_step_single_life
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
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        score_weight = 900000
        penalty=0
        score = score*params['score']
        high_score = high_score*params['high_score']
        death = -1*(death*params['death'])
        moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        death_no_food = -1*(death_no_food*params['death_no_food'])
        fit = high_score+death+moves_without_food+death_no_food+penalty

        return fit

    def score_death(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()

        score = score*params['score']
        same_dir_as_before = -1*(params['same_dir_as_before']*(same_dir_as_before))
        moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        death = -1*(death*params['death'])
        if death > 0:
            -1*(death/score+1)
        fit = 0
        fit=fit+score+death
        return fit


    def move_no_food(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        death = -1*(death*params[death])
        moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        fit=fit+moves_without_food
        return fit


    def train_death(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        death = -1*(death*params['death'])
        same_dir_as_before = -1*(same_dir_as_before*params['same_dir_as_before'])
        fit=fit+moves_without_food+death
        return fit


    def kejitech(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0

        death_no_food = -1*(death_no_food*params['death_no_food'])
        high_score = high_score*params['high_score']
        moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        death = -1*(death*params['death'])
        fit=fit+moves_without_food+death+high_score+death_no_food
        return fit

    def food_death(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0
        # high_score = high_score*params['high_score']
        # moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        # death = -1*(death*params['death'])
        fit = score / params['moves_without_food']
        return fit

    def food_death(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0
        # high_score = high_score*params['high_score']
        # moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        death = -1*(death*params['death'])
        fit = death+(score / params['moves_without_food'])

        return fit

    def food_death_simple(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0
        score = score*params['score']
        # moves_without_food = moves_without_food*params['moves_without_food']
        death = -1*(death*params['death'])
        fit = score + death
        return fit
    def death_simple(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0
        # score = score*params['score']
        # moves_without_food = moves_without_food*params['moves_without_food']
        death = -1*(death*params['death'])
        fit = death
        return fit
    def food_step_single_life(self, result):
        params = self.params
        high_score, step, score, death, death_no_food, exploration,moves_without_food, same_dir_as_before, moves = result.values()
        fit = 0

        moves_without_food = -1*(moves_without_food*params['moves_without_food'])
        fit = death
        return fit
