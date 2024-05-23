#!/usr/bin/env python


from datetime import datetime
import os
import random
from Fitness import Fitness
from Result import Result
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
from collections import Counter
import matplotlib.pyplot as plt

# dims = (7, 9, 15, 3)


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1, max_steps_in_game=1000,verbose=True, dims = (9, 9, 15, 3), fitness = None):
        self.population_size = population_size
        self.generations = generations
        self.population = []
        self.keep_ratio=keep_ratio
        self.mutation = mutation
        self.models = []
        self.gen_info = []
        self.verbose = verbose
        self.max_steps_in_game = max_steps_in_game
        self.dims = dims
        self.fitness = fitness
        self.max_data = []

    def __repr__(self) -> str:
        return (f"GeneticAlgorithm(population_size={self.population_size}, generations={self.generations}, "
                f"keep_ratio={self.keep_ratio}, mutation={self.mutation}, max_steps_in_game={self.max_steps_in_game}, "
                f"verbose={self.verbose}, dims={self.dims}, fitness={self.fitness})")

    def initialize_population(self):

        for _ in range(self.population_size):
            game = SnakeGame()
            controller = GAController(game=game,dims=self.dims)
            self.models.append(controller.model)
        print(len(self.models))


    def evolve(self):

        print(len(self.models))
        for gen in range(self.generations):
            population = []
            for model in self.models:
                high_score = 0
                result = {"high_score":0,
                          "step":0,
                          "score":0,
                          "death":0,
                          "death_no_food":0,
                          "exploration":0,
                          "moves_without_food":0,
                          "same_dir_as_before":0,
                          "moves":{"right":0,"left":0,"up":0,"down":0}}
                while result['step'] < self.max_steps_in_game:
                    game = SnakeGame(accum_step=result['step'],
                                     max_steps_in_game=self.max_steps_in_game)
                    controller = GAController(game=game, model=model, dims=self.dims, fitness_function=self.fitness)
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
                    result['same_dir_as_before'] += game.same_dir_as_before
                    # Increment steps in each direction
                    result['moves'] = dict(Counter(result['moves']) + Counter(controller.moves))
                controller.result = result
                population.append(controller)
            self.gen_info.append(population)
            population = self.rank_fitness(population)
            self.save_info(population)
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

    def generate_plots(self, data):
        x_values = [t[0] for t in data]
        y_values = [t[1] for t in data]
        z_values = [t[2] for t in data]

        # Create a figure and three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

        # Plot for the first values
        ax1.plot(x_values, marker='o', linestyle='-')
        ax1.set_title('Max Fitness')
        ax1.set_xlabel('Fitness')
        ax1.set_ylabel('Generations')

        # Plot for the second values
        ax2.plot(y_values, marker='o', linestyle='-')
        ax2.set_title('Max Apples Eaten Score')
        ax2.set_xlabel('Generations')
        ax2.set_ylabel('Score')

        # Plot for the third values
        ax3.plot(z_values, marker='o', linestyle='-')
        ax3.set_title('Plot of Third Values')
        ax3.set_xlabel('Generations')
        ax3.set_ylabel('Death')

        # Adjust layout to prevent overlap
        plt.tight_layout()

        # Ensure the 'plots' directory exists in the root directory
        plot_directory = os.path.join(os.getcwd(), 'plots')
        os.makedirs(plot_directory, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save the plot to a file in the 'plots' directory
        plot_path = os.path.join(plot_directory, f'plot_{timestamp}.png')
        plt.savefig(plot_path)

        # Show the plots
        # plt.show()

        print(f"Plot saved to {plot_path}")

    # saves info from a sorted array
    def save_info(self, population):
        max_fitness = population[0].fitness
        max_score = population[0].result['score']
        max_death = population[0].result['death']
        # return as tupple
        self.max_data.append((max_fitness, max_score, max_death,))

    def rank_fitness(self, population):
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


class TestGenerator:
    def __init__(self, iterations=0):
        self.algo_settings = algo_settings
        self.fitness_settings = fitness_settings
        self.iterations = iterations
        self.algo_setting_arr = []
        self.fitness_setting_arr = []

    def modify_settings(self, base_settings, attribute_changes, setting_type):

        for i in range(1, self.iterations + 1):
            new_settings = base_settings.copy()  # Create a new copy of base settings in each iteration
            print(base_settings)
            ii = 0
            for change in attribute_changes:
                attr_name = change["name"]
                attr_value = change["value"] * i
                if isinstance(new_settings[attr_name], int):
                    new_settings[attr_name] += attr_value
            if setting_type == "algo":
                self.algo_setting_arr.append(new_settings)
            elif setting_type == "fitness":
                self.fitness_setting_arr.append(new_settings)
    def combine(self):
        for algo, fitness in zip(self.algo_setting_arr,self.fitness_setting_arr):
            algo['fitness'] = fitness

    def run(self):
        for setting in self.algo_setting_arr:
            print(setting)




if __name__ == '__main__':
    # algo_settings = {
    #     'population_size':60,
    #     'generations': 50,
    #     'keep_ration':0.25,
    #     'mutation':0.08,
    #     'max_steps_in_game':700,
    #     'dims':(7,9,15,3),
    #     'fitness': {},
    #     'verbose': False
    # }
    algo_settings = {
        'population_size':60,
        'generations': 50,
        'keep_ratio':0.25,
        'mutation':0.08,
        'max_steps_in_game':700,
        'dims':(7,9,15,3),
        'fitness': {},
        'verbose': False
    }
    fitness_settings = {
        'name':'score_death',
        'same_dir_as_befire': -0.05,
        'score': 1000,
        'death':50
    }

    algo_attr = [{"name":"population_size","value":0},{"name":"generations","value":0}]
    fitness_attr = [{"name":"death","value":10}]
    tests = TestGenerator(iterations=20)
    #Iterate over the settings attributes defined in algo_attr and increment value
    #with the specified values in the dict
    tests.modify_settings(algo_settings,algo_attr,"algo")
    tests.modify_settings(fitness_settings,fitness_attr,"fitness")
    tests.combine()
    tests.run()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder = f'{fitness_settings['name']}_{timestamp}'
    for settings in tests.algo_setting_arr:
        fitness = Fitness(method=settings['fitness']['name'],params=settings['fitness'])
        ga=GeneticAlgorithm(population_size=settings['population_size'],
                            generations=settings['generations'],
                            keep_ratio=settings['keep_ratio'],
                            mutation=settings['mutation'],
                            max_steps_in_game=settings['max_steps_in_game'],
                            dims=(7, 9, 15, 3),
                            fitness=fitness,
                            verbose=False)

        ga.initialize_population()
        ga.evolve()
        ga.print_generation_fitness(ga.gen_info)
        ga.final_result()
        ga.generate_plots(ga.max_data,)
