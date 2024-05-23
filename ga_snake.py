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

    def generate_plots(self, data, test_folder,iterator=""):
        fit_val = [t[0] for t in data]
        score_val = [t[1] for t in data]
        death_val = [t[2] for t in data]
        no_food_val = [t[3] for t in data]
        same_dir_val = [t[4] for t in data]

        # Create a figure and three subplots
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(8, 12))


        ax1.plot(fit_val, marker='o', linestyle='-')
        ax1.set_title('Max Fitness')
        ax1.set_xlabel('Generations')
        ax1.set_ylabel('Fitness')

        ax2.plot(score_val, marker='o', linestyle='-')
        ax2.set_title('Max Apples Eaten Score')
        ax2.set_xlabel('Generations')
        ax2.set_ylabel('Score')

        ax3.plot(death_val, marker='o', linestyle='-')
        ax3.set_title('Max Deaths')
        ax3.set_xlabel('Generations')
        ax3.set_ylabel('Death')

        ax4.plot(no_food_val, marker='o', linestyle='-')
        ax4.set_title('Max Moves without food')
        ax4.set_xlabel('Generations')
        ax4.set_ylabel('Moves')

        ax5.plot(same_dir_val, marker='o', linestyle='-')
        ax5.set_title('Same direction as before')
        ax5.set_xlabel('Generations')
        ax5.set_ylabel('Moves')
        # Adjust layout to prevent overlap
        plt.tight_layout()
        folder_path = os.path.join(test_folder, f'{iterator}_plot')  # Path inside the test_folder
        os.makedirs(folder_path, exist_ok=True)
        plot_directory = folder_path

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Save the plot to a file in the 'plots' directory
        name = f'plot_{timestamp}.png'
        plot_path = os.path.join(plot_directory, name)
        plt.savefig(plot_path)

        # Show the plots
        # plt.show()

        print(f"Plot saved to {plot_path}")

    # saves info from a sorted array
    def save_info(self, population):
        max_fitness = population[0].fitness
        max_score = population[0].result['score']
        max_death = population[0].result['death']
        max_moves_without_food = population[0].result['moves_without_food']
        max_same_dir_as_before = population[0].result['same_dir_as_before']

        # return as tupple
        self.max_data.append((max_fitness, max_score, max_death,max_moves_without_food,max_same_dir_as_before,))

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
    def print_generation_fitness(self, generations, test_folder,iterator=""):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = f"{iterator}_generation_score"
        folder_path = os.path.join(test_folder, folder_name)

          # Path inside the test_folder
        os.makedirs(folder_path, exist_ok=True)
        file_name = os.path.join(folder_path, f"data_{timestamp}.txt")
        with open(file_name, 'w') as f:
            for i, pop in enumerate(generations, start=0):
                pop = self.rank_fitness(pop)
                f.write("##################################################\n")
                f.write(f"Generation: {i}\n")
                for j, controller in enumerate(pop, start=0):
                    f.write(f"Controller {j+1} Fitness = {controller.fitness}, death: {controller.result['death']}, step: {controller.result['step']}, score: {controller.result['score']}\n")
                    f.write(f"Moves: {controller.moves}\n")
                    f.write('____________________________________________________________\n')
                    if j > 0:
                        break
                f.write(f"LENGTH: {len(pop)}\n")

    def final_result(self,verbose):
        print("###########FINAL RESULT########")
        last_gen = self.rank_fitness(self.gen_info[-1])
        if verbose == True:
            self.print_fitness(last_gen)
        return last_gen

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

    def find_highest_scoring_population(self, data_tuple):
        highest_score = -float('inf')
        best_population = None

        for idx, data in enumerate(data_tuple):
            # We take index 0, because index 1 is the settings for that data
            ranked_population = sorted(data[0], key=lambda x: x.fitness, reverse=True)
            settings = data[1]
            iteration = data[2]
            highest_in_population = ranked_population[0].fitness

            print(f"Highest score in population {idx}: {highest_in_population}")

            if highest_in_population > highest_score:
                highest_score = highest_in_population
                best_population = ranked_population
                best_settings = settings

        return best_population, best_settings, iteration

    def save_result_txt(self, best_population, best_settings, iteration, test_folder,folder_name):
        folder_path = os.path.join(test_folder, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        file_path = os.path.join(folder_path, 'best_test.txt')

        with open(file_path, 'w') as f:
            f.write("Best Test Result:\n")
            f.write(f"Iteration (0 based): {iteration} \n")
            # f.write(f"Highest Score: {highest_score}\n")
            f.write("Best Settings:\n")
            for key, value in best_settings.items():
                f.write(f"{key}: {value}\n")
            f.write("Best Population:\n")
            for i, controller in enumerate(best_population):
                f.write(f"Controller {i+1} Fitness: {controller.fitness}, Score: {controller.result['score']}, Deaths: {controller.result['death']}, Steps: {controller.result['step']}\n")
                print(f"Controller {i+1} Fitness: {controller.fitness}, Score: {controller.result['score']}, Deaths: {controller.result['death']}, Steps: {controller.result['step']}\n")

        print(f"Best test result saved to {file_path}")



def test_genetic_algo( iterations, algo_settings, algo_attr,fitness_settings,fitness_attr):
        tests = TestGenerator(iterations=iterations)
        #Iterate over the settings attributes defined in algo_attr and increment value
        #with the specified values in the dict
        tests.modify_settings(algo_settings,algo_attr,"algo")
        tests.modify_settings(fitness_settings,fitness_attr,"fitness")
        tests.combine()
        tests.run()

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


        folder = f'test_results/{fitness_settings["name"]}_{timestamp}/test'
        test_folder = os.path.join(os.getcwd(), folder)
        i=0
        data = []
        for settings in tests.algo_setting_arr:
            i+=1
            print("STARTING ALGORITHM NUMBER: ",i)
            fitness = Fitness(method=settings['fitness']['name'],params=settings['fitness'])
            ga=GeneticAlgorithm(population_size=settings['population_size'],
                                generations=settings['generations'],
                                keep_ratio=settings['keep_ratio'],
                                mutation=settings['mutation'],
                                max_steps_in_game=settings['max_steps_in_game'],
                                dims=settings['dims'],
                                fitness=fitness,
                                verbose=False)

            ga.initialize_population()
            ga.evolve()
            ga.print_generation_fitness(ga.gen_info,test_folder,i)
            # left , in the end to create a tupple
            final_result = ga.final_result(verbose=False)
            folder_name = f'{i}_settings'
            tests.save_result_txt(final_result,settings,i, test_folder, folder_name),
            data.append((final_result,settings,i,))
            ga.generate_plots(ga.max_data,test_folder,i)
        # find best among the test settings
        best_population, best_settings, iteration = tests.find_highest_scoring_population(data)
        tests.save_result_txt(best_population, best_settings, iteration, test_folder,f"0BEST_TEST")
        return (best_population, best_settings, iteration, folder,)


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
        'generations': 2,
        'keep_ratio':0.25,
        'mutation':0.08,
        'max_steps_in_game':700,
        'dims':(7,9,15,3),
        'fitness': {},

    }
    fitness_settings = {
        'name':'score_death',
        'same_dir_as_befire': -0.05,
        'score': 500,
        'death':50
    }
    fitness_settings_high_score = {
        'name':'high_score',
        'score': 1000,
        'high_score': 10000,
        'death': 100,
        'moves_without_food': 100,
        'death_no_food': 100,
    }


    final_result = []
    algo_attr = [{"name":"population_size","value":0},{"name":"generations","value":0}]
    # fitness attr
    fitness_attr = [{"name":"score","value":50}]
    # high score attr
    fitness_attr_high_score = [{"name":"moves_without_food","value":50}]
    result = test_genetic_algo(2, algo_settings, algo_attr, fitness_settings, fitness_attr)
    final_result.append(result)
    result2 = test_genetic_algo(2, algo_settings, algo_attr, fitness_settings_high_score, fitness_attr_high_score)
    final_result.append(result2)


    for idx, result in enumerate(final_result):
        print(f"Result of test {idx + 1}:")
        ranked_population = sorted(result[0], key=lambda x: x.fitness, reverse=True)
        print("Fitess score:",ranked_population[0].fitness,"Score: ",ranked_population[0].score)
        print(f"Settings: {result[1]}")
        print(f"Iteration Number in folder: {result[2]}")
        print(f"Folder name: {result[3]}\n")