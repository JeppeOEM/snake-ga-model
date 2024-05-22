#!/usr/bin/env python


from datetime import datetime
import os
import random
from Result import Result
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
from collections import Counter
dims = (7, 9, 15, 3)


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1, max_steps_in_game=1000,verbose=True):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.keep_ratio=keep_ratio
        self.mutation = mutation
        self.models = []
        self.gen_info = []
        self.verbose = verbose
        self.max_steps_in_game = max_steps_in_game

    def initialize_population(self):

        for _ in range(self.population_size):
            game = SnakeGame()
            controller = GAController(game)
            self.models.append(controller.model)
        print(len(self.models))


    def evolve(self):

        print(len(self.models))
        for gen in range(self.generations):
            population = []
            for model in self.models:
                high_score = 0
                result = {"high_score":0,"step":0, "score":0,"death":0,"death_no_food":0,"exploration":0,"moves_without_food":0, "moves":{"right":0,"left":0,"up":0,"down":0}}
                while result['step'] < self.max_steps_in_game:
                    game = SnakeGame(accum_step=result['step'],
                                     max_steps_in_game=self.max_steps_in_game)
                    controller = GAController(game, model)
                    game.run()
                    if game.score > high_score:
                        high_score = game.score
                        result['high_score'] = high_score
                    result['step'] += game.step
                    result['score'] += game.score
                    result['death'] += game.death
                    result['death_no_food'] += game.death_no_food
                    result['exploration'] += game.exploration
                    result['moves_without_food'] += game.snake.moves_without_food
                    # Increment steps in each direction
                    result['moves'] = dict(Counter(result['moves']) + Counter(controller.moves))
                controller.result = result
                population.append(controller)
            self.gen_info.append(population)
            parents = self.selection(population)
            babies = self.mate_in_pairs(parents)
            parent_models = self.extract_models(parents)
            self.models = babies + parent_models # combine arrays
            print("Generation:",gen)
            print(f"{parents[1].fitness} {parents[1].result['score']}")
            for key , value in parents[1].result.items():
                print(key,value)





            # For debugging
            # self.print_edge("first",new_pop)
    def selection(self, population):
        population = self.rank_fitness(population)
        if self.verbose:
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
    def print_this(self, i, controller):
        return f"Controller {i} #| Fitness = {controller.fitness} |# "
    def print_fitness(self, pop):

        for i, controller in enumerate(pop, start=0):
            print("___________________________________________________________")
            print(self.print_this(i, controller))
            print(controller.result)
            print("___________________________________________________________")
            if i > 10:
                break
        print("LENGHT : ",len(pop))
    def print_generation_fitness(self, generations):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = "generation_score"
        os.makedirs(folder_name, exist_ok=True)
        file_name = os.path.join(folder_name, f"generation_score_{timestamp}.txt")
        with open(file_name, 'w') as f:
            for i, pop in enumerate(generations, start=0):
                pop = self.rank_fitness(pop)
                f.write("##################################################\n")
                f.write(f"Generation: {i}\n")
                for j, controller in enumerate(pop, start=0):
                    f.write(f"Controller {j+1} Fitness = {controller.fitness}, death: {controller.result['death']}, step: {controller.result['step']}, score: {controller.result['score']}\n")
                    f.write(f"Moves: {controller.moves}\n")
                    f.write('____________________________________________________________\n')
                    if j > 4:
                        break
                f.write(f"LENGHT: {len(pop)}\n")
    def final_result(self):
        print("###########FINAL RESULT########")
        self.print_fitness(self.rank_fitness(self.gen_info[-1]))
if __name__ == '__main__':
    ga=GeneticAlgorithm(population_size=60,
                        generations=200,
                        keep_ratio=0.25,
                        mutation=0.08,
                        max_steps_in_game=700,
                        verbose=False)
    ga.initialize_population()
    ga.evolve()
    ga.print_generation_fitness(ga.gen_info)
    ga.final_result()
