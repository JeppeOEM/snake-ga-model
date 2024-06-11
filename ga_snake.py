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


if __name__ == '__main__':
    final_result = []



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


    kejitech = {
        'name':'kejitech',
        'death_no_food': 1000,
        'moves_without_food': 100,
        'death': 150,
        'high_score': 5000
    }

    food_death_simple = {
        'name':'food_death_simple',
        'score': 100,
        'death': 500,
    }


    food_death_simple2 = {
        'name':'food_death_simple',
        'score': 500,
        'death': 100,
    }

    kejitech = {
        'name':'kejitech',
        'death_no_food': 1000,
        'moves_without_food': 100,
        'death': 150,
        'high_score': 5000
    }

    death_simple = {
        'name':'death_simple',
        'death': 150,
    }

    food_step = {
        'name':'food_step',
        'score': 100,
        'moves_without_food':0.1,
    }
    score_step = {
        'name':'food_step',
        'score': 200,
        'moves_without_food':0.3,
    }


    algo_settings_basic = {
        'population_size':8000,
        'generations': 120,
        'keep_ratio':0.01,
        'mutation':0.05,
        'max_steps_in_game':700000,
        'dims':(9,9,15,3),
        'fitness': {},
        'seed_fitness':{}
    }

    algo_settings_dir = {
        'population_size':8000,
        'generations': 120,
        'keep_ratio':0.05,
        'mutation':0.15,
        'max_steps_in_game':700000,
        'dims':(7,9,15,3),
        'fitness': {},
        'seed_fitness':{}
    }

    algo_settings = {
        'population_size':8000,
        'generations': 320,
        'keep_ratio':0.01,
        'mutation':0.1,
        'max_steps_in_game':700000,
        'dims':(4,9,15,3),
        'fitness': {},
        'seed_fitness':{}
    }


    algo_attr = [{"name":"population_size","value":0},{"name":"generations","value":0}]
    attr_same = [{"name":"same_dir_as_before","value":2}]
    attr_food = [{"name":"moves_without_food","value":4}]
    attr_score = [{"name":"score","value":100}]
    seed_attr = [{'name':'death',"value": 0}]
    kejitech_attr = [{'name':'high_score',"value":5000}]
    food_step_attr = [{'name':'moves_without_food',"value":0.2},{'name':'score',"value":0}]
    score_step_attr = [{'name':'moves_without_food',"value":0.2},{'name':'score',"value":0}]


    result1 = test_genetic_algo(False, # verbose.. prints exta info
                                iterations=1, #Iterations genetic algo will run with new settings
                                algo_settings=algo_settings,
                                algo_attr=algo_attr, # attributes to increase each iteration
                                fitness_settings=score_step,
                                fitness_attr=score_step_attr,
                                obs="default") # attributes to increas each iteration

    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             iterations=1, #Iterations genetic algo will run with new settings
    #                             algo_settings=algo_settings_basic,
    #                             algo_attr=algo_attr, # attributes to increase each iteration
    #                             fitness_settings=score_step,
    #                             fitness_attr=score_step_attr,
    #                             obs="basic") # attributes to increas each iteration


    # result1 = test_genetic_algo(False, # verbose.. prints exta info
    #                             iterations=1, #Iterations genetic algo will run with new settings
    #                             algo_settings=algo_settings_dir,
    #                             algo_attr=algo_attr, # attributes to increase each iteration
    #                             fitness_settings=score_step,
    #                             fitness_attr=score_step_attr,
    #                             obs="direction") # attributes to increas each iteration



    for idx, result in enumerate(final_result):
        print(f"Result of test {idx + 1}:")
        ranked_population = sorted(result[0], key=lambda x: x.fitness, reverse=True)
        print("Fitess score:",ranked_population[0].fitness,"Score: ",ranked_population[0].score)
        print(f"Settings: {result[1]}")
        print(f"Iteration Number in folder: {result[2]}")
        print(f"Folder name: {result[3]}\n")



