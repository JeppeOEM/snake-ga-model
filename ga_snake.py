#!/usr/bin/env python


import random
from Result import Result
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
from collections import Counter
dims = (7, 9, 15, 3)


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.keep_ratio=keep_ratio
        self.mutation = mutation
        self.models = []
        self.gen_info = []

    def initialize_population(self):

        for _ in range(self.population_size):
            game = SnakeGame()
            controller = GAController(game)
            game.run()
            self.models.append(controller.model)
        print(len(self.models))


    def evolve(self):

        print(len(self.models))
        for gen in range(self.generations):
            population = []
            for model in self.models:
                result = {"step":0, "score":0,"death":0,"without_food":0, "moves":{"right":0,"left":0,"up":0,"down":0}}
                while result['step'] < 50:
                    game = SnakeGame()
                    controller = GAController(game, model)
                    game.run()
                    result['step'] += game.step
                    result['score'] += game.score
                    result['death'] += game.death
                    result['without_food'] += game.without_food
                    # Increment steps in each direction
                    result['moves'] = dict(Counter(result['moves']) + Counter(controller.moves))
                controller.result = result
                population.append(controller)
            self.gen_info.append(population)
            parents = self.selection(population)
            babies = self.mate_in_pairs(parents)
            parent_models = self.extract_models(parents)
            self.models = babies + parent_models # combine arrays





            # For debugging
            # self.print_edge("first",new_pop)
    def selection(self, population):
        population = self.rank_fitness(population)
        self.print_fitness(population)
        top_peformers = int(len(population) * self.keep_ratio)
        parents = population[:top_peformers] # remove worst slices to the end of list
        return parents
    def extract_models(self, parents):
        models = []
        for parent in parents:
             models.append(parent.model)
        return models
    def mate_in_pairs(self, population):
        print("Lenght:",len(population))
        babies = []
        size = self.population_size - len(population)

        while len(babies) < size:
                dad, mom = random.sample(population, 2)
                baby = dad.model + mom.model
                baby2 = dad.model + mom.model
                baby.mutate(self.mutation)
                baby2.mutate(self.mutation)
                babies.append(baby)
                babies.append(baby2)

        return babies


    def rank_fitness(self, population):
            population = sorted(population, key=lambda x: x.fitness, reverse=True)
            return population
    def print_fitness(self, pop):
        for i, controller in enumerate(pop, start=0):
            print(controller.result)
            print(f"Controller {i+1} death:{controller.result['death']} :step:{controller.result['step']} score {controller.result['score']} Fitness = {controller.fitness} same move total ")
        print("LENGHT : ",len(pop))
    def print_generation_fitness(self, generations):
        for i, pop in enumerate(generations, start=0):
            print("Generation:",i)
            for i, controller in enumerate(pop, start=0):
                print(controller.result)
                print(f"Controller {i+1} death:{controller.result['death']} :step:{controller.result['step']} score {controller.result['score']} Fitness = {controller.fitness} same move total ")
                if i > 5:
                    break
            print("LENGHT : ",len(pop))

if __name__ == '__main__':
    ga=GeneticAlgorithm(population_size=50,generations=10,keep_ratio=0.2,mutation=0.05)
    ga.initialize_population()
    ga.evolve()
    ga.print_generation_fitness(ga.gen_info)
