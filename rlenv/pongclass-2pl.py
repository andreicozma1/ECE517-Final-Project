import time

import pygame
import math
import numpy as np
import random
from typing import Union

# pong game class
class pongGame:
    # initializing parameters
    def __init__(self, h, w, draw=True, draw_speed: Union[float, None] = None):
        # window height
        self.h = h
        # window width
        self.w = w
        # If you intend to use for visualization, set draw to True
        self.should_draw = draw
        self.draw_speed = draw_speed
        self.reset()
        if self.should_draw:
            pygame.init()
            self.window = pygame.display.set_mode((self.h, self.w))
            self.draw()

    def reset_ball(self):
        # ball x and y location
        self.xball = self.w / 2
        self.yball = self.h / 2
        # ball speed and angle
        self.angle = random.random() * 0.5 * math.pi + 0.75 * math.pi
        self.totalSpeed = 2
        self.ballHDirection = self.totalSpeed * math.cos(self.angle)
        self.ballVDirection = self.totalSpeed * math.sin(self.angle)

    def reset(self):
        self.reset_ball()
        # self.last_action = np.finfo(np.float32).eps.item()
        # player paddle location
        self.y1 = self.h / 2 - 40
        # computer paddle location
        self.y2 = self.h / 2
        # paddle length
        self.paddle_length = self.h / 6

        self.last_action_p1 = 2
        self.last_action_p2 = 2

    # returns all the parameters for the game (player location, computer location, x of ball, y of ball, x direction of ball, y direction of ball)
    def getState(self):
        return np.array([self.y1, self.y2, self.xball, self.yball, self.ballHDirection, self.ballVDirection],
                        dtype=np.float32)
        # return np.array([self.y1, self.xball, self.yball],
        #                 dtype=np.float32)
        # return np.array([self.y1, self.xball, self.yball, self.ballHDirection, self.ballVDirection],
        #                 dtype=np.float32)
        # return np.array([self.y1, self.y2, self.xball, self.yball, self.ballHDirection, self.ballVDirection,
        #                  self.last_action],
        #                 dtype=np.float32)

        # return np.array([self.y1, self.xball, self.yball], dtype=np.float32)

    # Take one step of the game
    def takeAction(self, p1_action, p2_action, p1_ai=False, p2_ai=True):
        self.last_action_p1 = p1_action
        self.last_action_p2 = p2_action
        # reward
        p1_r = 0
        p2_r = 0
        # move action (up is 0, down is 1, no move is 2)
        if p1_action == 0 and self.y1 > 5:
            self.y1 = self.y1 - 5
        elif p1_action == 1 and self.y1 < self.w - self.paddle_length - 5:
            self.y1 = self.y1 + 5

        if p2_action == 0 and self.y2 > 5:
            self.y2 = self.y2 - 5
        elif p2_action == 1 and self.y2 < self.w - self.paddle_length - 5:
            self.y2 = self.y2 + 5

        if p1_ai:
            # also move the player paddle on its own
            if self.yball > self.y1 + self.paddle_length / 2:
                self.y1 = self.y1 + 5
            elif self.yball < self.y1 + self.paddle_length / 2:
                self.y1 = self.y1 - 5

        if p2_ai:
            # move computer paddle on its own
            if self.yball > self.y2 + self.paddle_length / 2:
                self.y2 = self.y2 + 5
            elif self.yball < self.y2 + self.paddle_length / 2:
                self.y2 = self.y2 - 5

        # math for when paddle hits ball

        if self.y1 < self.yball < self.y1 + self.paddle_length and 0 < self.xball < 15:
            self.totalSpeed = self.totalSpeed + 0.2
            paddle_middle = self.paddle_length / 2
            self.angle = (math.pi / 4) * (self.yball - (self.y1 + self.paddle_length / 2)) / paddle_middle
            self.xball = 15
            p1_r += 1
        if self.y2 < self.yball < self.y2 + self.paddle_length and self.w - 15 < self.xball < self.w:
            self.totalSpeed = self.totalSpeed + 0.2
            paddle_middle = self.paddle_length / 2
            self.angle = math.pi - (math.pi / 4) * (self.yball - (self.y2 + self.paddle_length / 2)) / paddle_middle
            self.xball = self.w - 15
            p2_r += 1

        # if p1 lose, p2 win
        if self.xball < 0:
            p1_r = -100
            p2_r = 100
        # if p1 win, p2 lose
        if self.xball > self.w:
            p1_r = 100
            p2_r = -100

        # if ball hits top or bottom wall
        if self.yball <= 0 or self.yball >= self.h:
            self.angle = -self.angle

        # recalculate ball location
        self.ballHDirection = self.totalSpeed * math.cos(self.angle)
        self.ballVDirection = self.totalSpeed * math.sin(self.angle)
        self.xball = self.xball + self.ballHDirection
        self.yball = self.yball + self.ballVDirection

        self.draw()
        # return reward
        return p1_r, p2_r

    # a function to draw the actual game frame. If called it is probably best to use a delay between frames (for example time.sleep(0.03) for about 30 frames per second.
    def draw(self):
        if not self.should_draw:
            return
        if self.draw_speed is not None:
            time.sleep(self.draw_speed)
        # clear the display
        self.window.fill(0)

        paddle_width = 10
        paddle_distance = 5

        # draw the scene
        pygame.draw.rect(self.window, (255, 255, 255), (paddle_distance, self.y1,
                                                        paddle_width, self.paddle_length))
        pygame.draw.rect(self.window, (255, 255, 255), (self.w - paddle_width - paddle_distance, self.y2,
                                                        paddle_width, self.paddle_length))
        pygame.draw.circle(self.window, (255, 255, 255), (self.xball, self.yball), 5)
        # update the display
        pygame.display.flip()
