import math
from typing import Protocol, Tuple

import numpy as np
from ga_models.ga_simple import SimpleModel
from vector import Vector
import pygame
from game_controller import GameController


class GAController(GameController):
    def __init__(self, game=None, model=None,display=True, dims=None, fitness_function=None):
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
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, moves = result.values()
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
        # moves_with_out_food = -100*moves_with_out_food
        # exploration = (exploration / death) * score
        # print(death)
        # fit = 0
        # if score > 0:
        #     fit = step/(score*score_weight)

        # fit = fit+death+moves_with_out_food
        # fit = fit+death_no_food
        score = score*1000
        high_score = high_score*100
        death = -1*(death*100)
        moves_with_out_food = -1*(moves_with_out_food*30)
        death_no_food = -1*(death_no_food*100)
        # print(death_no_food)

        fit = high_score+death+moves_with_out_food+death_no_food+penalty

        return fit

    # @property
    # def fitness(self):
        high_score, step, score, death, death_no_food, exploration,moves_with_out_food, moves = self.result.values()
        score_weight = 900000
        penalty=0
        total_moves = sum(moves.values())
        percentage_moves = {key: (value / total_moves) * 100 for key, value in moves.items()}
        for direction, percentage in percentage_moves.items():
            if percentage > 55:
                penalty = -(900000+death)


        # exploration = exploration * 0.001
        # exploration = (exploration / death)
        # death_no_food = -1000*(death_no_food)
        # death = -150*(death)
        # moves_with_out_food = -100*moves_with_out_food
        # exploration = (exploration / death) * score
        # print(death)
        # fit = 0
        # if score > 0:
        #     fit = step/(score*score_weight)

        # fit = fit+death+moves_with_out_food
        # fit = fit+death_no_food
        # score = score*1000
        high_score = high_score*10000
        death = -1*(death*150)
        moves_with_out_food = -1*(moves_with_out_food*100)
        death_no_food = -1*(death_no_food*1000)
        # print(death_no_food)

        fit = high_score+death+moves_with_out_food+death_no_food+penalty

        return fit

    def eucludian(self, apple_position, snake_position):
        return ((apple_position.x - snake_position.x)**2 + (apple_position.y - snake_position.y)**2)**0.5

    def angle_with_apple(self, snake_position, apple_position):
        apple_vector = np.array(apple_position.x) - np.array(snake_position.y)
        snake_vector = np.array(snake_position.x) - np.array(snake_position.y)

        norm_apple_vector = np.linalg.norm(apple_vector)
        norm_snake_vector = np.linalg.norm(snake_vector)

        apple_normalized = apple_vector / norm_apple_vector
        snake_normalized = snake_vector / norm_snake_vector
        # Calculate the angle between the snake's direction and the vector to the apple
        normalized_angle = np.dot([snake_position.x,snake_position.y], [apple_position.x,apple_position.y]) / (np.linalg.norm([snake_position.x,snake_position.y]) * np.linalg.norm([apple_position.x,apple_position.y]))

        try:
            apple_normalized = apple_vector / norm_apple_vector
            snake_normalized = snake_vector / norm_snake_vector
            normalized_angle = np.dot([snake_position.x,snake_position.y], [apple_position.x,apple_position.y]) / (np.linalg.norm([snake_position.x,snake_position.y]) * np.linalg.norm([apple_position.x,apple_position.y]))
        except ZeroDivisionError:
            print(" zero encountered when normalizing vectors. I Init vectors with (0.0) ")
            # Handle the error gracefully, such as setting default values or logging the error
            apple_normalized = np.zeros_like(apple_vector)
            snake_normalized = np.zeros_like(snake_vector)
            normalized_angle = 0  # Or any other suitable value

        return normalized_angle, apple_normalized, snake_normalized


    def update(self) -> Vector:
        # observation space

        # delta north, east, south, west
        dn = self.game.snake.p.y
        de = self.game.grid.x - self.game.snake.p.x
        ds = self.game.grid.y - self.game.snake.p.y
        dw = self.game.snake.p.x
        max_distance = max(self.game.grid.x - 1, self.game.grid.y - 1)  # Maximum possible distance in the grid
        dn = dn / max_distance
        de = de / max_distance
        ds = ds / max_distance
        dw = dw / max_distance
        # # delta food x and y
        # dfx = self.game.snake.p.x - self.game.food.p.x
        # dfy = self.game.snake.p.y - self.game.food.p.y

        self.eucludian(self.game.food.p, self.game.snake.p)
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
            angle,norm_apple_vector, norm_snake_vector =self.angle_with_apple(self.game.food.p, last_move)
            # print("angle",angle,"apple",norm_apple_vector,"snake", norm_snake_vector)
           #gets threats from the diffrent direction possible left,right,straight
            danger = self.calc_direction(last_move)
            threat_right = danger[0]
            threat_left = danger[1]
            threat_straight = danger[2]
            normalized_snake_x = last_move.x
            normalized_snake_y = last_move.y
            # print(last_move.x)

        # last_move = self.game.snake.last_move
        # if last_move is not None:
        #     if last_move == Vector(0, -1):  # Last move was up
        #         self.action_space = (Vector(-1, 0), Vector(1, 0), Vector(0, -1))  # Left, right, straight
        #         self.moves['up'] += 1
        #         # print("last move up")
        #     elif last_move == Vector(0, 1):  # Last move was down
        #         self.action_space = (Vector(1, 0), Vector(-1, 0), Vector(0, 1))  # Right, left, straight
        #         self.moves['down'] += 1
        #         # print("last move down")
        #     elif last_move == Vector(-1, 0):  # Last move was left
        #         self.moves['left'] += 1
        #         self.action_space = (Vector(0, -1), Vector(0, 1), Vector(-1, 0))  # Straight, up, down
        #         # print("last move left")
        #     elif last_move == Vector(1, 0):  # Last move was right
        #         self.action_space = (Vector(0, 1), Vector(0, -1), Vector(1, 0))  # Straight, down, up
        #         self.moves['right'] += 1
        #         # print("last move right")

            # Calculate and normalize Euclidean distance to food

    # # Threats from borders: 1 if next to border, 0 otherwise

        # normatnlized_dn = dn / (self.game.grid.y - 1)
        # normalized_de = de / (self.game.grid.x - 1)
        # normalized_ds = ds / (self.game.grid.y - 1)
        # normalized_dw = dw / (self.game.grid.x - 1)
        obs = (dn, de, ds, dw, norm_apple_vector, threat_left,threat_right,threat_straight, s)
        print(obs)
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
            self.clock.tick(30)
        return next_move
    def calc_direction(self, last_move):
            if last_move == Vector(0, -1):  # Last move was up
                self.action_space = (Vector(-1, 0), Vector(1, 0), Vector(0, -1))  # Left, right, straight
                self.moves['up'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.x == 0 else 0
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.x == self.game.grid.x - 1 else 0
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.y == 0 else 0
                # There are no threats in the left and right directions because it's moving up
                return [border_left,border_right,border_straight]

            elif last_move == Vector(0, 1):  # Last move was down
                self.action_space = (Vector(1, 0), Vector(-1, 0), Vector(0, 1))  # Right, left, straight
                self.moves['down'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.x == self.game.grid.x - 1 else 0
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.x == 0 else 0
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.y == self.game.grid.y - 1 else 0
                # There are no threats in the left and right directions because it's moving down
                return [border_left,border_right,border_straight]

            elif last_move == Vector(-1, 0):  # Last move was left
                self.moves['left'] += 1
                self.action_space = (Vector(0, -1), Vector(0, 1), Vector(-1, 0))  # Straight, up, down
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.y == 0 else 0
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.y == self.game.grid.y - 1 else 0
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.x == 0 else 0
                # There are no threats in the left and right directions because it's moving left
                return [border_left,border_right,border_straight]

            elif last_move == Vector(1, 0):  # Last move was right
                self.action_space = (Vector(0, 1), Vector(0, -1), Vector(1, 0))  # Straight, down, up
                self.moves['right'] += 1
                # Check if left move is next to a border
                border_left = 1 if self.game.snake.p.y == self.game.grid.y - 1 else 0
                # Check if right move is next to a border
                border_right = 1 if self.game.snake.p.y == 0 else 0
                # Check if straight move is next to a border
                border_straight = 1 if self.game.snake.p.x == self.game.grid.x - 1 else 0
                # There are no threats in the left and right directions because it's moving right
                return [border_left,border_right,border_straight]

    def block(self, obj):
        return (obj.x * self.game.scale,
                obj.y * self.game.scale,
                self.game.scale,
                self.game.scale)

    def calculate_valid_moves(self) -> Tuple[Vector, ...]:

        # Calculate valid moves based on the current state of the snake.

        if self.game.snake.last_move is None:
            return ()  # Return an empty tuple if last move is None
        else:
            last_move = self.game.snake.last_move

        # Define valid moves based on the last move
        valid_moves = ()

        # Check if moving up is valid
        if last_move != Vector(0, -1):
            move_up = Vector(0, 1)
            valid_moves += (move_up,)

        # Check if moving down is valid
        if last_move != Vector(0, 1):
            move_down = Vector(0, -1)
            valid_moves += (move_down,)

        # Check if moving left is valid
        if last_move != Vector(1, 0):
            move_left = Vector(-1, 0)
            valid_moves += (move_left,)

        # Check if moving right is valid
        if last_move != Vector(-1, 0):
            move_right = Vector(1, 0)
            valid_moves += (move_right,)

        return valid_moves

    def __str__(self):
        return f"__STR__:GAController(food={self.game.food},food={self.game.snake}, display={self.display})"
