
from datetime import datetime
import os
import pprint

from Fitness import Fitness
from GeneticAlgorithm import GeneticAlgorithm


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


TEST_RUN=False

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

            init_pop = ga.initialize_population()
            ga.evolve(init_pop)
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
        tests.save_result_txt(best_population, best_settings, iteration, test_folder,f"0_BEST_TEST")
        return (best_population, best_settings, iteration, folder,)

