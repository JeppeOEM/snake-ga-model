from datetime import datetime
import os
import pprint
import random
from Fitness import Fitness
from ga_models.ga_simple import SimpleModel
from snake import SnakeGame
from ga_controller import GAController
from collections import Counter
import matplotlib.pyplot as plt

from test_genetic_algo import test_genetic_algo

# dims = (7, 9, 15, 3)




if __name__ == '__main__':
    final_result = []

    algo_settings = {
        'population_size':2000,
        'generations': 100,
        'keep_ratio':0.5,
        'mutation':0.06,
        'max_steps_in_game':700,
        'dims':(4,9,15,3),
        'fitness': {},
        'seed_fitness':{}
    }

    algo_settings_small = {
        'population_size':2000,
        'generations': 100,
        'keep_ratio':0.5,
        'mutation':0.06,
        'max_steps_in_game':700,
        'dims':(4,6,3),
        'fitness': {},
        'seed_fitness':{}
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

    fitness_food_death_step_simple = {
        'name':'food_death_step_simple',
        'score': 100,
        'death': 500,
        'moves_without_food':0.1,
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

    fitness_death_simple = {
        'name':'death_simple',
        'death': 150,
    }


    algo_attr = [{"name":"population_size","value":0},{"name":"generations","value":0}]
    fitness_attr_same = [{"name":"same_dir_as_before","value":2}]
    fitness_attr_food = [{"name":"moves_without_food","value":4}]
    fitness_attr_score = [{"name":"score","value":100}]
    fitness_seed_attr = [{'name':'death',"value": 0}]
    fitness_kejitech_attr = [{'name':'high_score',"value":5000}]
    fitness_food_death_simple_attr = [{'name':'death',"value":0},{'name':'score',"value":50}]
    fitness_food_death_step_simple_attr = [{'name':'moves_without_food',"value":0.2},{'name':'score',"value":0}]

    result1 = test_genetic_algo(False, # verbose.. prints exta info
                                1, #Iterations genetic algo will run with new settings
                                algo_settings,
                                algo_attr, # attributes to increase each iteration
                                fitness_kejitech,
                                fitness_seed_attr) # attributes to increas each iteration
    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             1, #Iterations genetic algo will run with new settings
    #                             algo_settings,
    #                             algo_attr, # attributes to increase each iteration
    #                             fitness_kejitech,
    #                             fitness_seed_attr) # attributes to increas each iteration
    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             1, #Iterations genetic algo will run with new settings
    #                             algo_settings_small,
    #                             algo_attr, # attributes to increase each iteration
    #                             fitness_kejitech,
    #                             fitness_seed_attr) # attributes to increas each iteration
    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             3, #Iterations genetic algo will run with new settings
    #                             algo_settings,
    #                             algo_attr, # attributes to increase each iteration
    #                             fitness_food_death_step_simple,
    #                             fitness_food_death_step_simple) # attributes to increas each iteration
    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             3, #Iterations genetic algo will run with new settings
    #                             algo_settings_small,
    #                             algo_attr, # attributes to increase each iteration
    #                             fitness_food_death_step_simple,
    #                             fitness_food_death_step_simple) # attributes to increas each iteration


    final_result.append(result1)


    for idx, result in enumerate(final_result):
        print(f"Result of test {idx + 1}:")
        ranked_population = sorted(result[0], key=lambda x: x.fitness, reverse=True)
        print("Fitess score:",ranked_population[0].fitness,"Score: ",ranked_population[0].score)
        print(f"Settings: {result[1]}")
        print(f"Iteration Number in folder: {result[2]}")
        print(f"Folder name: {result[3]}\n")




    # algo_settings_dense_nn = {
    #     'population_size':60,
    #     'generations': 150,
    #     'keep_ratio':0.25,
    #     'mutation':0.06,
    #     'max_steps_in_game':700,
    #     'dims':(7,120,120,3),
    #     'fitness': {},
    #     'seed_fitness':{}
    # }

    # fitness_settings_score_death = {
    #     'name':'score_death',
    #     'same_dir_as_before': 0.5,
    #     'moves_without_food': 1,
    #     'score': 500,
    #     'death':50
    # }

    # settings_train_death = {
    #     'name':'train death',
    #     'same_dir_as_before': 0.5,
    #     'death': 200,
    # }