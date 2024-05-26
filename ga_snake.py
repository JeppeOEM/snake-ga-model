#!/usr/bin/env python


from datetime import datetime
import os
import pprint
import random
from Fitness import Fitness
from Result import Result
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
from collections import Counter
import matplotlib.pyplot as plt

# dims = (7, 9, 15, 3)
TEST_RUN=False


class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1, max_steps_in_game=1000,verbose=True, dims = (9, 9, 15, 3), fitness = None, fitness_seed = None, change_fitness=0):
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
        self.fitness_seed = fitness_seed
        self.change_fitness = change_fitness # what generation should fitness function be swapped
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
        # print(len(self.models))


    def evolve(self):
        max_fitness_so_far = float('-inf')
        print("Evolving a new generation")
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
                    if gen < self.change_fitness:
                        print("CHANGING FITNESS FUNCTION")
                        controller = GAController(game=game, model=model, dims=self.dims, fitness_function=self.fitness_seed)
                    else:
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
            if parents[0].fitness > max_fitness_so_far:
                max_fitness_so_far = parents[0]
            print(f"Generation {gen + 1}/{self.generations}: Best Fitness = {parents[0].fitness} Score = {parents[0].fitness} Highest Fitness So Far = {max_fitness_so_far.fitness} Score: {max_fitness_so_far.score}", end='\r', flush=True)
            if gen == self.generations:
                print("Last Generation:",gen)
                print(f"{parents[0].fitness} {parents[0].result['score']}")
                for key , value in parents[0].result.items():
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
            # Create and save the additional plot for fitness data only
        fig_fitness, ax_fitness = plt.subplots(figsize=(8, 6))

        ax_fitness.plot(fit_val, marker='o', linestyle='-')
        ax_fitness.set_title('Max Fitness')
        ax_fitness.set_xlabel('Generations')
        ax_fitness.set_ylabel('Fitness')

        fitness_plot_name = f'fitness_plot_{timestamp}.png'
        fitness_plot_path = os.path.join(folder_path, fitness_plot_name)
        plt.savefig(fitness_plot_path)

        # Show the plots
        # plt.show()

        print(f"Plots saved to {plot_path}")

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
        self.algo_settings = None
        self.fitness_settings = None
        self.iterations = iterations
        self.algo_setting_arr = []
        self.fitness_setting_arr = []
        self.fitness_setting_seed_arr = []

    def modify_settings(self, base_settings, attribute_changes, setting_type):
        for i in range(self.iterations):
            new_settings = base_settings.copy()  # Create a new copy of base settings in each iteration
            for change in attribute_changes:
                attr_name = change["name"]
                attr_value = change["value"] * i
                if attr_name in new_settings and isinstance(new_settings[attr_name], (int, float)):
                    new_settings[attr_name] += attr_value
            if setting_type == "algo":
                self.algo_setting_arr.append(new_settings)
            elif setting_type == "fitness":
                self.fitness_setting_arr.append(new_settings)
            elif setting_type == "fitness_seed":
                self.fitness_setting_seed_arr.append(new_settings)

    def combine(self):
        for i in range(self.iterations):
            if i < len(self.fitness_setting_arr):
                self.algo_setting_arr[i]['fitness'] = self.fitness_setting_arr[i]
            if i < len(self.fitness_setting_seed_arr):
                self.algo_setting_arr[i]['fitness_seed'] = self.fitness_setting_seed_arr[i]

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
                if i == 0:
                    print(f"Controller {i+1} Fitness: {controller.fitness}, Score: {controller.result['score']}, Deaths: {controller.result['death']}, Steps: {controller.result['step']}\n")

        print(f"Best test result saved to {file_path}")



def test_genetic_algo(verbose,
                      iterations,
                      algo_settings,
                      algo_attr,
                      fitness_settings,
                      fitness_attr,
                      fitness_seed=None,
                      fitness_seed_attr=None,
                      change_fitness=0):
        if TEST_RUN:
            iterations = 3
            algo_settings['generations'] = 3
            print("******************TEST RUN******************")
            print("SET TEST_RUN = False TO DISABLE")

        tests = TestGenerator(iterations=iterations)
        #Iterate over the settings attributes defined in algo_attr and increment value
        #with the specified values in the dict
        tests.modify_settings(algo_settings,algo_attr,"algo")
        tests.modify_settings(fitness_settings,fitness_attr,"fitness")
        if fitness_seed != None:
            tests.modify_settings(fitness_settings,fitness_seed_attr,"fitness_seed")
        tests.combine()

        if TEST_RUN:
            print("Testing if different settings are generated")
            pp = pprint.PrettyPrinter(indent=4)  # Create a PrettyPrinter instance with an indentation of 4 spaces
            for algo_settings in tests.algo_setting_arr:
                print("Algorithm Settings:")
                pp.pprint(algo_settings)


        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = f'test_results/{timestamp}/{fitness_settings["name"]}_{timestamp}/'
        test_folder = os.path.join(os.getcwd(), folder)
        i=0
        data = []
        for settings in tests.algo_setting_arr:
            i+=1
            fitness = Fitness(method=settings['fitness']['name'],params=settings['fitness'])
            if fitness_seed != None:
                fitness_seed = Fitness(method=settings['fitness_seed']['name'],params=settings['fitness_seed'])
            print("STARTING ALGORITHM NUMBER: ",i)

            ga=GeneticAlgorithm(population_size=settings['population_size'],
                                generations=settings['generations'],
                                keep_ratio=settings['keep_ratio'],
                                mutation=settings['mutation'],
                                max_steps_in_game=settings['max_steps_in_game'],
                                dims=settings['dims'],
                                fitness=fitness,
                                fitness_seed=fitness_seed,
                                change_fitness=change_fitness,
                                verbose=False)

            ga.initialize_population()
            ga.evolve()
            if verbose == True:
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
    TEST_RUN = False
    final_result = []

    algo_settings = {
        'population_size':60,
        'generations': 200,
        'keep_ratio':0.25,
        'mutation':0.06,
        'max_steps_in_game':700,
        'dims':(8,9,15,3),
        'fitness': {},
        'seed_fitness':{}
    }
    algo_settings_dense_nn = {
        'population_size':60,
        'generations': 150,
        'keep_ratio':0.25,
        'mutation':0.06,
        'max_steps_in_game':700,
        'dims':(8,120,120,3),
        'fitness': {},
        'seed_fitness':{}
    }

    fitness_settings_score_death = {
        'name':'score_death',
        'same_dir_as_before': 0.5,
        'moves_without_food': 1,
        'score': 500,
        'death':50
    }

    settings_train_death = {
        'name':'train death',
        'same_dir_as_before': 0.5,
        'death': 200,
    }
    fitness_kejitech = {
        'name':'kejitech',
        'death_no_food': 1000,
        'moves_without_food': 100,
        'death': 150,
        'high_score': 5000
    }

    fitness_food_death_simple = {
        'name':'food_death_simple',
        'score': 100,
        'death': 500,
    }
    fitness_food_death_simple2 = {
        'name':'food_death_simple',
        'score': 500,
        'death': 100,
    }
    fitness_kejitech = {
        'name':'kejitech',
        'death_no_food': 1000,
        'moves_without_food': 100,
        'death': 150,
        'high_score': 5000
    }



    algo_attr = [{"name":"population_size","value":0},{"name":"generations","value":0}]
    fitness_attr_same = [{"name":"same_dir_as_before","value":2}]
    fitness_attr_food = [{"name":"moves_without_food","value":4}]
    fitness_attr_score = [{"name":"score","value":100}]
    fitness_seed_attr = [{'name':'death',"value": 0}]
    fitness_kejitech_attr = [{'name':'high_score',"value":5000}]
    fitness_food_death_simple_attr = [{'name':'death',"value":0},{'name':'score',"value":50}]
    fitness_food_death_simple_attr2 = [{'name':'death',"value":50},{'name':'score',"value":0}]




    result1 = test_genetic_algo(False, # verbose.. prints exta info
                                6, #Iterations genetic algo will run with new settings
                                algo_settings,
                                algo_attr, # attributes to increase each iteration
                                fitness_food_death_simple,
                                fitness_food_death_simple_attr) # attributes to increas each iteration
    result2 = test_genetic_algo(False, # verbose.. prints exta info
                                6, #Iterations genetic algo will run with new settings
                                algo_settings,
                                algo_attr, # attributes to increase each iteration
                                fitness_food_death_simple2,
                                fitness_food_death_simple_attr2) # attributes to increas each iteration

    # result1 = test_genetic_algo(2,
    #                             algo_settings,
    #                             algo_attr,
    #                             fitness_settings_score_death,
    #                             fitness_attr_same,
    #                             settings_train_death,
    #                             fitness_seed_attr,
    #                             100) # number of generations to run before chaning fitness function

    # result1 = test_genetic_algo(2,
    #                             algo_settings,
    #                             algo_attr,
    #                             fitness_settings_score_death,
    #                             fitness_attr_same,
    #                             settings_train_death,
    #                             fitness_seed_attr,
    #                             100) # number of generations to run before chaning fitness function

    # result2 = test_genetic_algo(2,
    #                             algo_settings,
    #                             algo_attr,
    #                             fitness_settings_score_death,
    #                             fitness_attr_food,
    #                             settings_train_death,
    #                             fitness_seed_attr,
    #                             100) # number of generations to run before chaning fitness function

    # result3 = test_genetic_algo(10,
    #                             algo_settings,
    #                             algo_attr,
    #                             fitness_settings_score_death,
    #                             fitness_attr_food,
    #                             settings_train_death,
    #                             fitness_seed_attr,
    #                             100) # number of generations to run before chaning fitness function

    final_result.append(result1)


    for idx, result in enumerate(final_result):
        print(f"Result of test {idx + 1}:")
        ranked_population = sorted(result[0], key=lambda x: x.fitness, reverse=True)
        print("Fitess score:",ranked_population[0].fitness,"Score: ",ranked_population[0].score)
        print(f"Settings: {result[1]}")
        print(f"Iteration Number in folder: {result[2]}")
        print(f"Folder name: {result[3]}\n")
    if TEST_RUN == True:
        print("RUNNING AS TEST RUN")
        print("SET TEST_RUN = False TO DISABLE")