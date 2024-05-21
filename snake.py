import random
from collections import deque
from typing import Protocol
import pygame
from vector import Vector
from game_controller import HumanController


class SnakeGame:
    def __init__(self, xsize: int=30, ysize: int=30, scale: int=15, result=None):
        self.grid = Vector(xsize, ysize)
        self.scale = scale
        self.snake = Snake(game=self)
        self.food = Food(game=self)
        self.step = 0
        self.score = 0
        self.death = 0
        self.result = result

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
            self.step += 1
            if not self.snake.p.within(self.grid):
                running = False
                message = 'Game over! You crashed into the wall!'
                self.score += 1
                self.score += 1
                self.score += 1
            if self.snake.cross_own_tail:
                running = False
                message = 'Game over! You hit your own tail!'
                self.score += 1
                self.score += 1
                self.score += 1
            if self.snake.p == self.food.p:
                self.snake.add_score()
                self.food = Food(game=self)
                self.score += 1
        print(f'{message} ... Score: {self.snake.score}')


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
        self.body.append(Vector.random_within(self.game.grid))
        self.last_move = Vector(0, 0)

    def move(self):
        self.p = self.p + self.v
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
