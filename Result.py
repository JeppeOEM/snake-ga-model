class Result:
    def __init__(self):
        self.step = 0
        self.death = 0
        self.score = 0

    def __repr__(self):
        return f"Result(step={self.step}, death={self.death}, score={self.score})"
