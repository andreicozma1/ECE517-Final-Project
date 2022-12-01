import time

import pygame
import math
import numpy as np
import random
from typing import Union

PI_4 = math.pi / 4


# pong game class
class pongGame:
    # initializing parameters
    def __init__(self, h, w, draw=True, draw_speed: Union[float, None] = None):
        self.win_w, self.win_h = w, h
        self.draw_enabled, self.draw_speed = draw, draw_speed
        ####################################################################
        # Paddle width and height
        self.p_w, self.p_h = 10, self.win_h / 6
        # Distance from edge of screen to paddle
        self.p_dist_edge = 5
        # Collision distance for ball from paddle
        self.p_coll_dist = self.p_w + self.p_dist_edge
        # Paddle movement speed
        self.p_vel = 5

        ####################################################################
        self.p1_y, self.p2_y = None, None
        self.p1_last_action = None

        ####################################################################
        self.b_x, self.b_y = None, None
        self.b_angle, self.b_vel_h, self.b_vel_y = None, None, None
        self.b_speed, self.b_speed_increase = None, 0.2

        ####################################################################
        self.reset()
        if self.draw_enabled:
            pygame.init()
            self.window = pygame.display.set_mode((self.win_w, self.win_h))
            self.draw()

    def reset(self):
        self.p1_y, self.p2_y = self.win_h / 2 - 40, self.win_h / 2
        self.p1_last_action = 2
        self.reset_ball()

    def reset_ball(self):
        self.b_x, self.b_y = self.win_w / 2, self.win_h / 2
        self.b_speed = 2
        self.b_angle = random.random() * 0.5 * math.pi + 0.75 * math.pi
        self.b_vel_h = self.b_speed * math.cos(self.b_angle)
        self.b_vel_y = self.b_speed * math.sin(self.b_angle)

    # returns all the parameters for the game
    # (player location, computer location,
    # x of ball, y of ball,
    # x direction of ball, y direction of ball)
    def getState(self, include_opponent=True, include_angle=True, include_direction=True, include_last_action=True):
        # Original state
        # state = np.array([self.y1, self.y2,
        #                   self.xball, self.yball,
        #                   self.ballHDirection, self.ballVDirection])
        state = [self.p1_y]
        if include_opponent:
            state.append(self.p2_y)
        state.extend((self.b_x, self.b_y))
        if include_angle:
            state.append(self.b_angle)
        if include_direction:
            state.extend((self.b_vel_h, self.b_vel_y))
        if include_last_action:
            state.append(self.p1_last_action)
        return np.array(state, dtype=np.float32)

    # Take one step of the game
    def takeAction(self, action,
                   ai_p1=False, ai_p2=True,
                   reward_step=0,
                   reward_hit=1,
                   reward_win=100, reward_lose=-100,
                   ):
        self.p1_last_action = action
        # reward
        reward = reward_step
        # move action (up is 0, down is 1, no move is 2)
        direction = 0 if action == 2 else -1 if action == 0 else 1
        p1_y = self.p1_y + direction * self.p_vel
        if p1_y >= 0 and p1_y + self.p_h <= self.win_h:
            self.p1_y = p1_y

        if ai_p1:
            if self.b_y > self.p1_y + self.p_h / 2:
                self.p1_y = self.p1_y + 5
            elif self.b_y < self.p1_y + self.p_h / 2:
                self.p1_y = self.p1_y - 5

        if ai_p2:
            if self.b_y > self.p2_y + self.p_h / 2:
                self.p2_y = self.p2_y + 5
            elif self.b_y < self.p2_y + self.p_h / 2:
                self.p2_y = self.p2_y - 5

        # math for when paddle hits ball
        p1_match_y = self.p1_y < self.b_y < self.p1_y + self.p_h
        p1_match_x = 0 < self.b_x < self.p_coll_dist
        p2_match_y = self.p2_y < self.b_y < self.p2_y + self.p_h
        p2_match_x = self.win_w - self.p_coll_dist < self.b_x < self.win_w
        if p1_match_x and p1_match_y:
            self.increase_speed()
            self.b_angle = (PI_4 * (self.b_y - (self.p1_y + self.p_h / 2)) / (self.p_h / 2))
            self.b_x = self.p_coll_dist
            reward += reward_hit
        elif p2_match_x and p2_match_y:
            self.increase_speed()
            self.b_angle = math.pi - PI_4 * (self.b_y - (self.p2_y + self.p_h / 2)) / (
                    self.p_h / 2)
            self.b_x = self.win_w - self.p_coll_dist

        # if you lose
        if self.b_x < 0:
            reward = reward_lose
        elif self.b_x > self.win_w:
            reward = reward_win

        # if ball hits top or bottom wall
        if self.b_y <= 0 or self.b_y >= self.win_h:
            self.b_angle = -self.b_angle

        # recalculate ball location
        self.b_vel_h = self.b_speed * math.cos(self.b_angle)
        self.b_vel_y = self.b_speed * math.sin(self.b_angle)
        self.b_x = self.b_x + self.b_vel_h
        self.b_y = self.b_y + self.b_vel_y

        self.draw()
        return reward

    def increase_speed(self):
        self.b_speed = self.b_speed + self.b_speed_increase

    # a function to draw the actual game frame.
    # If called it is probably best to use a delay between frames
    # (for example time.sleep(0.03) for about 30 frames per second.
    def draw(self):
        if not self.draw_enabled:
            return
        if self.draw_speed is not None:
            time.sleep(self.draw_speed)
        # clear the display
        self.window.fill(0)

        # draw the scene
        pygame.draw.rect(self.window,
                         (255, 255, 255),
                         (self.p_dist_edge,  # x
                          self.p1_y,  # y
                          self.p_w,  # width
                          self.p_h))  # height
        pygame.draw.rect(self.window,
                         (255, 255, 255),
                         (self.win_w - self.p_w - self.p_dist_edge,  # x
                          self.p2_y,  # y
                          self.p_w,  # width
                          self.p_h))  # height
        pygame.draw.circle(self.window,
                           (255, 255, 255),
                           (self.b_x, self.b_y), 5)  # ((x, y), radius)
        # update the display
        pygame.display.flip()
