import os
from typing import Optional

import numpy as np
import random
import queue
import sys

import gymnasium as gym
from gym import spaces



class Snake(gym.Env):
    '''
    Snake Game Environment inheriting from OpenAI's Gym environment class

    ## Description
    The game starts with a snake of length 2, it's Head is designated as H, it's Tail is designated as T.
    The snake is placed on a square grid with dimensions taken in as 'grid_dim' in the class's constructor.
    The snake can move it's Head up,down,left and right, it's body follows the Head. There is always one Apple 
    in a random position, designated as A, which enlarges the snake when eaten. The snake's goal is to collect
    as many apples as possible with it's Head.

    ## Arguments
    - `grid_dim`: the game's square grid dimensions
    - `max_step` (optional): maximum number of steps the snake can take before the game ends
    - `render_mode` (optional): mode of display (currently only 'ascii' available)

    ## Action Space
    Action Space is discrete, it contains 4 values, which symbolise the snake's movement directions:
    0 - up, 1 - down, 2 - left, 3 - right

    ## Observation Space
    Observation Space is an array of size grid_dim^2 of discrete values 0-4, symbolising different things present on the grid:
    0 - nothing, 1 - apple, 2 - snake head, 3 - snake tail, 4 - snake body.
    The array contains the first row of the grid, then the second, etc.

    ## Rewards:
    - ate an apple: +2
    - went into a wall: -1
    - made an illegal move (360 degree turn): -1

    ## Episode End
    The episode ends if the following happens:

    1. The snake walks into it's body or the wall.
    2. The snake makes an illegal move.
    3. The snake made more moves than 'max_steps' (if 'max_steps' was set in the constructor).

    '''

    def __init__(self, grid_dim, max_steps: Optional[int] = None, render_mode: Optional[str] = None):
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([5 for _ in range(grid_dim*grid_dim)])
        self.render_mode = render_mode
        self.grid_dim = grid_dim
        self.max_steps = max_steps
        if (max_steps == None):
            self.max_steps = sys.maxsize

    def reset(self):
        self.grid = [0 for _ in range(self.grid_dim*self.grid_dim)]
        center_field = self.grid_dim*(self.grid_dim//2) + (self.grid_dim//2) 
        self.grid[center_field] = 2
        self.grid[center_field - 1] = 3
        self.head_loc = center_field
        self.tail_loc = center_field - 1
        self.tail_queue = queue.SimpleQueue()
        self.tail_queue.put(self.head_loc)
        self.last_move = -1
        self.apples_eaten = 0


        apple_loc = random.randrange(len(self.grid))
        while(self.grid[apple_loc] == 2 or self.grid[apple_loc] == 3):
            apple_loc = random.randrange(len(self.grid))
        self.grid[apple_loc] = 1

        self.current_step = 0
        return self._get_obs()
    
    def _get_obs(self):
            observation = [0 for _ in range(len(self.grid))]
            for i in range(len(self.grid)):
                observation[i] = self.grid[i]
                if observation[i] == 5:
                    observation[i] = 4
            return np.array(observation)
    
    def step(self, action):
        assert self.action_space.contains(action)

        self.current_step += 1

        if action == 0:
            new_loc = self.head_loc-self.grid_dim
            self.grid[self.head_loc] = 5
        elif action == 1:
            new_loc = self.head_loc+self.grid_dim
            self.grid[self.head_loc] = 5
        elif action == 2:
            new_loc = self.head_loc-1
            self.grid[self.head_loc] = 4
        elif action == 3:
            new_loc = self.head_loc+1
            self.grid[self.head_loc] = 4
        

        #snake went out of bounds
        if  new_loc < 0 or new_loc >= len(self.grid) or ((action == 2 or action == 3) and new_loc//self.grid_dim != self.head_loc//self.grid_dim):
            #print("debug1")
            terminated = True
            reward = -1

        #illegal move
        elif (self.last_move == 0 and action == 1) or (self.last_move == 1 and action == 0) or (self.last_move == 3 and action == 2) or (self.last_move == 3 and action == 2):
            terminated = True
            reward = -1

        #snake went into itself
        elif self.grid[new_loc] == 3 or self.grid[new_loc] == 4 or self.grid[new_loc] == 5:
            terminated = True
            reward = 0


        else:
            terminated = False
            #if there is an appole in the new location
            if self.grid[new_loc] == 1:
                self.apples_eaten += 1
                reward = 2
                apple_loc = random.randrange(len(self.grid))
                #if all apples are eaten then end
                if self.apples_eaten >= len(self.grid)-2:
                    terminated = True
                #else create next apple
                else:
                    while (self.grid[apple_loc] != 0):
                        apple_loc = random.randrange(len(self.grid))
                    self.grid[apple_loc] = 1
            
            else:
                reward = 0
                self.grid[self.tail_loc] = 0
                self.tail_loc = self.tail_queue.get()
                self.grid[self.tail_loc] = 3
                

            self.head_loc = new_loc
            self.grid[self.head_loc] = 2
            self.tail_queue.put(new_loc)

        if self.current_step > self.max_steps:
            terminated = True

        return self._get_obs(), reward, terminated, {}

        

    
    def render(self):
        if self.render_mode == 'ascii' or self.render_mode == None:
            endline = '--'
            for _ in range(self.grid_dim):
                endline += '-'
            print(endline)
            for i in range(len(self.grid)):
                if i%self.grid_dim == 0:
                    print("|", end = '')
                if self.grid[i] == 0:
                    print(" ", end = '')
                elif self.grid[i] == 1:
                    print("A", end = '')
                elif self.grid[i] == 2:
                    print("H", end = '')
                elif self.grid[i] == 3:
                    print("T", end = '')
                elif self.grid[i] == 4:
                    print("-", end = '')
                elif self.grid[i] == 5:
                    print("|", end = '')
                if i%self.grid_dim == self.grid_dim-1:
                    print("|")
            print(endline)