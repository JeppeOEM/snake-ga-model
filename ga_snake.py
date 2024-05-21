#!/usr/bin/env python


from Result import Result
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
dims = (7, 9, 15, 3)


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.keep_ratio=keep_ratio
        self.mutation = mutation
        self.models = []

    def initialize_population(self):

        for _ in range(self.population_size):
            game = SnakeGame()
            controller = GAController(game)
            game.run()
            self.models.append(controller.model)
        print(len(self.models))


    def evolve(self):
        population = []

        for gen in range(self.generations):
            for model in self.models:
                step = 0
                result = {"step":0, "score":0,"death":0}
                while result['step'] < 50:
                    game = SnakeGame()
                    controller = GAController(game, model)
                    game.run()
                    result['step'] += game.step
                    result['score'] += game.score
                    result['death'] += game.score
                controller.result = result
                population.append(controller)
                print("population size",len(population))
            ranked = self.rank_fitness(population)
            self.print_fitness(ranked)




        self.print_fitness(ranked)
    def rank_fitness(self, population):
            population = sorted(population, key=lambda x: x.fitness, reverse=True)
            return population
    def print_fitness(self, pop):
        for i, controller in enumerate(pop, start=0):
            print(f"Controller {i+1} death:{controller.game.death} :step:{controller.game.step} score {controller.game.snake.score} Fitness = {controller.fitness} same move total ")
if __name__ == '__main__':
    ga=GeneticAlgorithm(population_size=10,generations=10,keep_ratio=0.2,mutation=0.05)
    ga.initialize_population()
    ga.evolve()
