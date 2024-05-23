import random
from collections import deque
from typing import Protocol
import pygame
from vector import Vector
from game_controller import HumanController


class SnakeGame:
    def __init__(self,
                 xsize: int=30,
                 ysize: int=30,
                 scale: int=15,
                 result=None,
                 accum_step=0,
                 max_steps_in_game=0):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        self.snake = Snake(game=self)
        self.food = Food(game=self)
        self.step = 0
        self.score = 0
        self.death = 0
        self.death_no_food = 0
        self.result = result
        self.exploration = 0
        self.accum_step = accum_step
        self.max_steps_in_game = max_steps_in_game
        self.same_dir_as_before = 0

    def add_to_result(self, value, target):

        current_value = getattr(self.result, target)
        print(current_value)
        setattr(self.result,target,current_value + value)

    def run(self):
        running = True
        while running:
            next_move = self.controller.update()
            if next_move: self.snake.v = next_move
            self.snake.move()

            self.snake.moves_without_food += 1
            self.step += 1
            if self.step + self.accum_step == self.max_steps_in_game:
                running = False
            if not self.snake.p.within(self.grid):
                running = False
                message = 'Game over! You crashed into the wall!'
                self.death += 1
                pygame.quit()
            if self.snake.cross_own_tail:
                running = False
                message = 'Game over! You hit your own tail!'
                self.death += 1
                pygame.quit()
            if self.snake.p == self.food.p:
                self.snake.add_score()
                self.food = Food(game=self)
                self.snake.moves_without_food = 0
                self.score += 1

            if self.snake.moves_without_food > self.snake.max_without_food:
                self.snake.score = 0
                running = False
                self.death += 1
                #death because of no food
                self.death_no_food +=1
                # self.snake.moves_without_food = 0
                pygame.quit()
                message = 'Game over! Took too many moves without eating!'
        # print(f'{message} ... Score: {self.snake.score}')


class Food:
    def __init__(self, game: SnakeGame):
        self.game = game
        self.p = Vector.random_within(self.game.grid)


class Snake:
    def __init__(self, *, game: SnakeGame):
        self.game = game
        self.score = 0
        self.v = Vector(0, 0)
        self.body = deque()
        # self.body.append(Vector.random_within(self.game.grid))
        self.last_move = Vector(0, 1)
        self.moves_without_food = 0
        self.max_without_food = 200
        initial_position = Vector.random_within(self.game.grid)
        self.body.append(initial_position)
        self.body.append(initial_position + Vector(1, 0))
    def same_direction(self):
        if self.v == self.last_move:
            self.game.same_dir_as_before +=1


    def move(self):
        self.p = self.p + self.v
        self.same_direction()
        self.last_move = self.v

    def get_last_move(self):
        return self.last_move

    @property
    def cross_own_tail(self):
        try:
            self.body.index(self.p, 1)
            return True
        except ValueError:
            return False

    @property
    def p(self):
        return self.body[0]

    @p.setter
    def p(self, value):
        self.body.appendleft(value)
        self.body.pop()

    def add_score(self):
        self.score += 1
        tail = self.body.pop()
        self.body.append(tail)
        self.body.append(tail)

    def debug(self):
        print('===')
        for i in self.body:
            print(str(i))
