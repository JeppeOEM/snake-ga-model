from collections import Counter
from datetime import datetime
import os
import random

from matplotlib import pyplot as plt

from ga_controller import GAController
from snake import SnakeGame

def print_object_ids(obj_list):
    ids = [id(obj) for obj in obj_list]
    print("Object IDs:", ids)
class GeneticAlgorithm:
    def __init__(self, population_size=10, generations=2,keep_ratio=0.1, mutation=0.1, max_steps_in_game=900000,verbose=True, dims = (9, 9, 15, 3), fitness = None, fitness_seed = None, change_fitness=0,obs="default"):
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
        self.obs = obs # what observational method to process the data inputs from the game



    def __repr__(self) -> str:
        return (f"GeneticAlgorithm(population_size={self.population_size}, generations={self.generations}, "
                f"keep_ratio={self.keep_ratio}, mutation={self.mutation}, max_steps_in_game={self.max_steps_in_game}, "
                f"verbose={self.verbose}, dims={self.dims}, fitness={self.fitness})")

    def initialize_population(self):
        models = []
        for _ in range(self.population_size):
            game = SnakeGame()
            controller = GAController(game=game,dims=self.dims,observation=self.obs)
            models.append(controller.model)
        return models



    def evolve(self, init_pop):
        max_fitness_so_far = float('-inf')
        print("Evolving a new generation")
        model_ga = init_pop
        for gen in range(self.generations):
            print("START")
            print(gen,  end='\r', flush=True)
            population = []
            for model in model_ga:
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
                # while result['step'] < self.max_steps_in_game:
                    game = SnakeGame(accum_step=result['step'],
                                     max_steps_in_game=self.max_steps_in_game)
             
                    controller = GAController(game=game, model=model, dims=self.dims, fitness_function=self.fitness, observation=self.obs)
                    if gen < self.change_fitness:
                        print("****Changing Fitness Function****")
                        controller = GAController(game=game, model=model, dims=self.dims, fitness_function=self.fitness_seed,observation=self.obs)
                    
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
                    result['moves'] = dict(Counter(result['moves']) + Counter(controller.moves))
                    if result['death'] > 0:
                        # print(result)
                        break
                    # Increment steps in each direction
                controller.result = result
                population.append(controller)
            self.gen_info.append(population)
            population = self.rank_fitness(population)
            self.save_info(population)
            parents = self.selection(population)
            babies = self.mate_in_pairs(parents)
            parent_model_ga = self.extract_models(parents)
            model_ga = []
            model_ga.extend(babies)
            model_ga.extend(parent_model_ga) # combine arrays

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
        extracted_models = []
        for parent in parents:
             extracted_models.append(parent.model)
        return extracted_models
    def mate_in_pairs(self, population):
        babies = []
        size = self.population_size - len(population)

        while len(babies) < size:
                dad, mom = random.sample(population, 2)
                baby = dad.model + mom.model
                baby.mutate(self.mutation)
                babies.append(baby)

        return babies

    def generate_plots(self, data, test_folder, iterator=""):
        fit_val = [t[0] for t in data]
        score_val = [t[1] for t in data]
        death_val = [t[2] for t in data]
        no_food_val = [t[3] for t in data]
        same_dir_val = [t[4] for t in data]

        plot_titles = ['Max Fitness', 'Max Apples Eaten Score', 'Max Deaths', 'Max Moves without Food', 'Same Direction as Before']
        plot_data = [fit_val, score_val, death_val, no_food_val, same_dir_val]
        y_labels = ['Fitness', 'Score', 'Death', 'Moves', 'Moves']

        folder_path = os.path.join(test_folder, f'{iterator}_plot')  # Path inside the test_results folder
        os.makedirs(folder_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        for i, (data, title, y_label) in enumerate(zip(plot_data, plot_titles, y_labels)):
            plt.figure(figsize=(8, 6))
            plt.plot(data, marker='o', linestyle='-')
            plt.title(title)
            plt.xlabel('Generations')
            plt.ylabel(y_label)
            plot_path = os.path.join(folder_path, f'{title.replace(" ", "_").lower()}_{timestamp}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Plot saved to {plot_path}")

    # saves info from a sorted array
    def save_info(self, population):
        max_fitness = population[0].fitness
        max_score = population[0].result['score']
        max_death = population[0].result['death']
        max_moves_without_food = population[0].result['moves_without_food']
        max_same_dir_as_before = population[0].result['same_dir_as_before']
        values = (max_fitness, max_score, max_death,max_moves_without_food,max_same_dir_as_before,)
        print(values)

        # return as tupple
        self.max_data.append(values)

    def rank_fitness(self, population):
        population = sorted(population, key=lambda x: x.fitness, reverse=True)
        

        for i, individual in enumerate(population):
            if i < 10:
                print(f"Individual {i}: Fitness = {individual.fitness} Score = {individual.result['score']}moves_without_food:{individual.result['moves_without_food']}")
    
        
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

