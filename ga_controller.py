import math
import pprint
from typing import Protocol, Tuple

import numpy as np
from ga_models.ga_simple import SimpleModel
from vector import Vector
import pygame
from game_controller import GameController


class GAController(GameController):
    def __init__(self, game=None, model=None,display=False, dims=None, fitness_function=None):
        self.display = display
        self.game = game
        self.model = model if model else SimpleModel(dims=dims) # type: ignore
        self.game.controller = self
        self.action_space = (Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0))
        self.death = 0
        self.step = 0
        self.score = 0
        self.result = {}
        self.moves = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
        self.fitness_function = fitness_function if fitness_function else self.default_fitness

        if self.display:
            pygame.init()
            self.screen = pygame.display.set_mode((game.grid.x * game.scale, game.grid.y * game.scale))
            self.clock = pygame.time.Clock()

            self.color_snake_head = (0, 255, 0)
            self.color_food = (255, 0, 0)
            self.action_space = (Vector(0, -1), Vector(0, 1), Vector(1, 0), Vector(-1, 0))

    def __del__(self):
        if self.display:
            pygame.quit()


    def set_game(self, game):
        self.game = game
        self.game.controller = game


    @property
    def fitness(self):
        return self.fitness_function(self.result)

    def default_fitness(self, result):
        high_score, step, score, death, death_no_food, exploration,moves_without_food, moves = result.values()
        score_weight = 900000
        penalty=0
        # total_moves = sum(moves.values())
        # percentage_moves = {key: (value / total_moves) * 100 for key, value in moves.items()}
        # for direction, percentage in percentage_moves.items():
        #     if percentage > 85:
        #         penalty = -1000


        # exploration = exploration * 0.001
        # exploration = (exploration / death)
        # death_no_food = -1000*(death_no_food)
        # death = -150*(death)
        # moves_without_food = -100*moves_without_food
        # exploration = (exploration / death) * score
        # print(death)
        # fit = 0
        # if score > 0:
        #     fit = step/(score*score_weight)

        # fit = fit+death+moves_without_food
        # fit = fit+death_no_food
        score = score*10000
        high_score = high_score*100000
        death = -2*(death*100)
        moves_without_food = -1*(moves_without_food*300)
        death_no_food = -1*(death_no_food*10000)
        # print(death_no_food)

        fit = high_score+death+moves_without_food+death_no_food+penalty

        return fit


    def eucludian(self, apple_position, snake_position):
        return ((apple_position.x - snake_position.x)**2 + (apple_position.y - snake_position.y)**2)**0.5

    def normalized_distance_to_food(self) -> float:
        # Calculate the Euclidean distance between the snake's head and the food
        distance = self.eucludian(self.game.food.p, self.game.snake.p)
        # Calculate the maximum possible distance in the grid (diagonal distance)
        max_distance = ((self.game.grid.x - 1) ** 2 + (self.game.grid.y - 1) ** 2) ** 0.5
        # Normalize the distance to a value between 0 and 1
        normalized_distance = distance / max_distance
        return normalized_distance


    def angle_with_apple(self):
        head = self.game.snake.body[0]
        after_head = self.game.snake.body[1]
        food = self.game.food.p
        # print(head, after_head, food)
        apple_postion = np.array([food.x, food.y]) - np.array([head.x, head.y])
        #head minus the body part behind gives direction
        snake_vector_dir = np.array([head.x, head.y]) - np.array([after_head.x, after_head.y])
        # print("apple postion",apple_postion, "snake_vector", snake_vector_dir)

        norm_apple_vector = np.linalg.norm(apple_postion)
        norm_snake_vector = np.linalg.norm(snake_vector_dir)
        if norm_apple_vector == 0:
            norm_apple_vector = 1
        if norm_snake_vector == 0:
            norm_snake_vector = 1
        # print("norm", norm_apple_vector, "normsnake", norm_snake_vector)
        apple_normalized = apple_postion / norm_apple_vector
        snake_normalized = snake_vector_dir / norm_snake_vector
        # print("apple norma", apple_normalized,"snake norm",snake_normalized)

        norm_of_apple_vector_dir = np.linalg.norm(apple_postion)
        norm_of_snake_vector_dir = np.linalg.norm(snake_vector_dir)
        # handle warning of 0 division on startup with a vector(0.0)
        if norm_of_apple_vector_dir == 0:
            norm_of_apple_vector_dir = 1
        if norm_of_snake_vector_dir == 0:
            norm_of_snake_vector_dir = 1

        # print("norm of", norm_of_apple_vector_dir, "normsnake of", norm_of_snake_vector_dir)
        apple_vector_dir_normalized = apple_postion / norm_of_apple_vector_dir
        snake_vector_dir_normalized = snake_vector_dir / norm_of_snake_vector_dir
        # print("norm of2222", apple_vector_dir_normalized, "normsnake o2222f", snake_vector_dir_normalized)

        angle = math.atan2(
            apple_vector_dir_normalized[1] * snake_vector_dir_normalized[0] - apple_vector_dir_normalized[
                0] * snake_vector_dir_normalized[1],
            apple_vector_dir_normalized[1] * snake_vector_dir_normalized[1] + apple_vector_dir_normalized[
                0] * snake_vector_dir_normalized[0]) / math.pi

        return angle, snake_vector_dir, apple_vector_dir_normalized, snake_vector_dir_normalized



        # Calculate the angle between the snake's direction and the vector to the apple
        # normalized_angle = np.dot([snake_position.x,snake_position.y], [apple_position.x,apple_position.y]) / (np.linalg.norm([snake_position.x,snake_position.y]) * np.linalg.norm([apple_position.x,apple_position.y]))

        # apple_normalized = apple_vector / norm_apple_vector
        # snake_normalized = snake_vector / norm_snake_vector
        # print("app norma",apple_normalized,"snake_norma",snake_normalized)
        # normalized_angle = np.dot([snake_position.x,snake_position.y], [apple_position.x,apple_position.y]) / (np.linalg.norm([snake_position.x,snake_position.y]) * np.linalg.norm([apple_position.x,apple_position.y]))
        # if np.isnan(normalized_angle) or np.any(np.isnan(apple_normalized)) or np.any(np.isnan(snake_normalized)):
        #     print("Warning: NaN detected in normalized_angle, apple_normalized, or snake_normalized")

        return normalized_angle, apple_normalized, snake_normalized


    def update(self) -> Vector:
        # observation space

        # delta north, east, south, west
        dn = self.game.snake.p.y
        de = self.game.grid.x - self.game.snake.p.x
        ds = self.game.grid.y - self.game.snake.p.y
        dw = self.game.snake.p.x
        # Normalized distance
        max_distance = max(self.game.grid.x - 1, self.game.grid.y - 1)  # Maximum possible distance in the grid
        dn = dn / max_distance
        de = de / max_distance
        ds = ds / max_distance
        dw = dw / max_distance
        # # delta food x and y
        # dfx = self.game.snake.p.x - self.game.food.p.x
        # dfy = self.game.snake.p.y - self.game.food.p.y
        # score
        s = self.game.snake.score

        # euclidean_distance_to_food = self.eucludian(self.game.food.p, self.game.snake.p)
        # max_distance = ((self.game.grid.x - 1) ** 2 + (self.game.grid.y - 1) ** 2) ** 0.5
        # dist_food = euclidean_distance_to_food / max_distance
        # print("norma",dist_food)
        # print(euclidean_distance_to_food)
        last_move = self.game.snake.last_move

        if last_move is not None:
            # print(last_move)
            angle, snake_vector_dir, apple_vector_dir_normalized, snake_vector_dir_normalized = self.angle_with_apple()
            #   


            # print(apple_vector_dir_normalized[0], snake_vector_dir_normalized[0], snake_vector_dir_normalized[1], snake_vector_dir_normalized[1])
            # print("angle",angle,"apple",norm_apple_vector,"snake", norm_snake_vector)
           #gets threats from the diffrent direction possible left,right,straight
            danger = self.calc_direction(last_move)
            threat_right = danger[0]
            threat_left = danger[1]
            threat_straight = danger[2]
            normalized_snake_x = last_move.x
            normalized_snake_y = last_move.y
            # print(last_move.x)

            # Calculate and normalize Euclidean distance to food

    # # Threats from borders: 1 if next to border, 0 otherwise

        # normatnlized_dn = dn / (self.game.grid.y - 1)
        # normalized_de = de / (self.game.grid.x - 1)
        # normalized_ds = ds / (self.game.grid.y - 1)
        # normalized_dw = dw / (self.game.grid.x - 1)

        normalized_dist_food = self.normalized_distance_to_food()
        # print(angle,normalized_dist_food)

        obs = (
            #    apple_vector_dir_normalized[0],
            #    apple_vector_dir_normalized[1],
            #    snake_vector_dir_normalized[0],
            #    snake_vector_dir_normalized[1],
               angle,
               threat_left,
               threat_right,
               threat_straight)
        # data = {
        #     # "norm dist food": normalized_dist_food,
        #     "apple vector dir 0": apple_vector_dir_normalized[0],
        #     "apple vector dir 1": apple_vector_dir_normalized[1],
        #     "snake vector dir 0": snake_vector_dir_normalized[0],
        #     "snake vector dir 1": snake_vector_dir_normalized[1],
        #     "left":threat_left,
        #     "right":threat_right,
        #     "straight":threat_straight
        # }
        # print(data)
        # pprint.pprint(data)
        # obs = (dn, de, ds, dw, dfx, dfy, tn,te,ts,tw, s)
        # obs = (dn, de, ds, dw, angle, norm_snake_vector, norm_apple_vector, threat_left,threat_right,threat_straight, s)
        # print(obs)
        # obs = (dn, de, ds, dw, dfx, dfy, s)

        # action space
        next_move = self.action_space[self.model.action(obs)]

        # display
        if self.display:
            self.screen.fill('black')
            for i, p in enumerate(self.game.snake.body):
                pygame.draw.rect(self.screen, (0, max(128, 255 - i * 12), 0), self.block(p))
            pygame.draw.rect(self.screen, self.color_food, self.block(self.game.food.p))
            pygame.display.flip()
            self.clock.tick(2)
            if self.step > 5:
                exit()
        return next_move
    def calc_direction(self, last_move):
            if last_move == Vector(0, -1):  # Last move was up
                self.action_space = (Vector(-1, 0), Vector(1, 0), Vector(0, -1))  # Left, right, straight
                self.moves['up'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.x == 0 else -1
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.x == self.game.grid.x - 1 else -1
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.y == 0 else -1
                # There are no threats in the left and right directions because it's moving up
                return [border_left,border_right,border_straight]

            elif last_move == Vector(0, 1):  # Last move was down
                self.action_space = (Vector(1, 0), Vector(-1, 0), Vector(0, 1))  # Right, left, straight
                self.moves['down'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.x == self.game.grid.x - 1 else -1
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.x == 0 else -1
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.y == self.game.grid.y - 1 else -1
                # There are no threats in the left and right directions because it's moving down
                return [border_left,border_right,border_straight]

            elif last_move == Vector(-1, 0):  # Last move was left
                self.moves['left'] += 1
                self.action_space = (Vector(0, -1), Vector(0, 1), Vector(-1, 0))  # Straight, up, down
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.y == 0 else -1
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.y == self.game.grid.y - 1 else -1
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.x == 0 else -1
                # There are no threats in the left and right directions because it's moving left
                return [border_left,border_right,border_straight]

            elif last_move == Vector(1, 0):  # Last move was right
                self.action_space = (Vector(0, 1), Vector(0, -1), Vector(1, 0))  # Straight, down, up
                self.moves['right'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.y == self.game.grid.y - 1 else -1
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.y == 0 else -1
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.x == self.game.grid.x - 1 else -1
                # There are no threats in the left and right directions because it's moving right
                return [border_left,border_right,border_straight]

    def block(self, obj):
        return (obj.x * self.game.scale,
                obj.y * self.game.scale,
                self.game.scale,
                self.game.scale)


    def __str__(self):
        return f"__STR__:GAController(food={self.game.food},food={self.game.snake}, display={self.display})"
