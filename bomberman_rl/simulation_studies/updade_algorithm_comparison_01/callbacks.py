import os
import pickle
import random

import numpy as np

import logging
import sys

import scipy.special

from .settings_train import *
from .settings_play import *

# Valid actions
ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT']

# Feature settings
NUM_LOOK_AROUND = 4

# Settings regarding the logging process
STDOUT_LOGLEVEL = logging.DEBUG

def setup(self):
    """
    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if STDOUT_LOGLEVEL != None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(STDOUT_LOGLEVEL)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    if self.train or not os.path.isfile("weights.pt"):
        self.logger.info(f"Setting up model from scratch (linear model with {get_num_features()} features).")
        self.weights = np.zeros((len(ACTIONS), get_num_features()))
    else:
        self.logger.info("Loading model from saved state.")
        with open("weights.pt", "rb") as file:
            self.weights = pickle.load(file)
            
def evaluate_q(features, action, weights):
    return np.dot(weights, features)[ACTIONS.index(action)]

# ------------------------------------------------------------------------------
# policy-manipulators ----------------------------------------------------------
# ------------------------------------------------------------------------------

def epsilon_greedy(q_values, epsilon):
    if random.random() < epsilon:
      return np.random.choice(len(q_values))
    else:
      return np.argmax(q_values)
  
def softmax(q_values, temperature):
    # naive implementation does not handle inf's in the numerator
    # prob = np.exp(q_values/temperature) / np.sum(np.exp(q_values/temperature))
    prob = scipy.special.softmax(q_values / temperature)
    return np.random.choice(len(q_values), p = prob)
    
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board. :return: The action to take as a string. """ 
    action_map, _ = normalize_state(game_state)
    features = state_to_features(game_state)
    q_values = np.dot(self.weights, features)
    
    action_index = None
    
    if self.train:
        if TRAIN_POLICY_TYPE == 'EPSILON-GREEDY':
            epsilon_train = np.interp(game_state['round'], EPSILON_TRAIN_BREAKS, EPSILON_TRAIN_VALUES)
            action_index = epsilon_greedy(q_values, epsilon_train)
        elif TRAIN_POLICY_TYPE == 'SOFTMAX':
            temperature_train = np.interp(game_state['round'], INVERSE_TEMPERATURE_TRAIN_BREAKS, 1/np.array(INVERSE_TEMPERATURE_TRAIN_VALUES))
            action_index = softmax(q_values, temperature_train)
        else:
            raise NotImplementedError(f"The policy type '{TRAIN_POLICY_TYPE}' is not implemented.")
    else:
        if PLAY_POLICY_TYPE == 'EPSILON-GREEDY':
            action_index = epsilon_greedy(q_values, EPSILON_PLAY)
        elif PLAY_POLICY_TYPE == 'SOFTMAX':
            action_index = softmax(q_values, 1/INVERSE_TEMPERATURE_PLAY)
        else:
            raise NotImplementedError(f"The policy type '{PLAY_POLICY_TYPE}' is not implemented.")
    
    return action_map(ACTIONS[action_index])
    
def normalize_state(game_state):
    """
    :param game_state: The dictionary that to normalize (in-place).
    
    :return: action_map: function to map action in normalized state to action in input_state,
    reverse_action_map: function to map action in input_state to action in normalized state.
    
    """
    if game_state == None:
        return lambda a: a, lambda a: a
    
    agent_x, agent_y = game_state['self'][3]
    
    def flip_tuple_x(t):
        return (16 - t[0], t[1])
        
    def flip_tuple_y(t):
        return (t[0], 16 - t[1])
   
    flipped_x = False
    if agent_x > 8:
        game_state['field'] = np.flipud(game_state['field'])
        game_state['bombs'] = [(flip_tuple_x(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.flipud(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_x(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, flip_tuple_x(pos))
        game_state['others'] = [(name, score, canPlaceBomb, flip_tuple_x(pos)) for name, score, canPlaceBomb, pos in game_state['others']]
        flipped_x = True

    flipped_y = False
    if agent_y > 8:
        game_state['field'] = np.fliplr(game_state['field'])
        game_state['bombs'] = [(flip_tuple_y(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.fliplr(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_y(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, flip_tuple_y(pos))
        game_state['others'] = [(name, score, canPlaceBomb, flip_tuple_y(pos)) for name, score, canPlaceBomb, pos in game_state['others']]
        flipped_y = True
        
    agent_x_update, agent_y_update = game_state['self'][3]
    
    def transpose_tuple(t):
        return (t[1], t[0])
    
    transposed_board = False
    if agent_y_update > agent_x_update:
        game_state['field'] = np.transpose(game_state['field'])
        game_state['bombs'] = [(transpose_tuple(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.transpose(game_state['explosion_map'])
        game_state['coins'] = [transpose_tuple(coin) for coin in game_state['coins']]
        name, score, canPlaceBomb, pos = game_state['self']
        game_state['self'] = (name, score, canPlaceBomb, transpose_tuple(pos))
        game_state['others'] = [(name, score, canPlaceBomb, transpose_tuple(pos)) for name, score, canPlaceBomb, pos in game_state['others']]
        transposed_board = True

    def transposed_action(a):
        mapping = {'RIGHT': 'DOWN', 'DOWN': 'RIGHT', 'LEFT': 'UP', 'UP': 'LEFT', 'WAIT': 'WAIT', 'BOMB': 'BOMB'}
        return mapping[a]
    
    def flipped_x_action(a):
        return 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        
    def flipped_y_action(a):
        return 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
    
    def action_map(a):
        a = transposed_action(a) if transposed_board else a
        a = flipped_x_action(a) if flipped_x else a
        a = flipped_y_action(a) if flipped_y else a
        return a
        
    def reverse_action_map(a):
        a = flipped_x_action(a) if flipped_x else a
        a = flipped_y_action(a) if flipped_y else a
        a = transposed_action(a) if transposed_board else a
        return a 
        
    return action_map, reverse_action_map

def get_num_features():
    dummy_state = {
      'round': 0,
      'step': 0,
      'field': np.zeros((17, 17)),
      'bombs': [],
      'explosion_map': np.zeros((17, 17)),
      'coins': [],
      'self': ("dummy", 0, True, (1, 1)),
      'others': []
    }
    
    return state_to_features(dummy_state).shape[0]
    
def state_to_features(game_state: dict, readable = False) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector. 
    :param game_state:  A dictionary describing the current game board.
    :return: np.array (NUM_FEATURES,)
    """
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
       
    self_x, self_y = game_state['self'][3]
    
    wall_map = np.zeros((31, 31)) 
    field = game_state['field'] 
    for x in np.arange(17):
        for y in np.arange(17):
            if (field[x, y] != -1):
                continue 
            
            x_rel, y_rel = x - self_x, y - self_y
            wall_map[15 + y_rel, 15 + x_rel] = 1
            
    coin_map = np.zeros((31, 31)) 
    coins = game_state['coins']
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[15 + y_rel, 15 + x_rel] = 1
        
    coins_in_quartal = [np.sum(coin_map[0:16,0:16]), np.sum(coin_map[0:16,16:32]), np.sum(coin_map[16:32,0:16]), np.sum(coin_map[16:32,16:32])]

    index_min = 15 - NUM_LOOK_AROUND
    index_max = 16 + NUM_LOOK_AROUND
    channels = [wall_map[index_min:index_max,index_min:index_max], coin_map[index_min:index_max,index_min:index_max]]

    max_coin_quartal = np.zeros(4) 
    max_coin_quartal[np.argmax(coins_in_quartal)] = 1

    
    if readable:
        return {
            "wall_map": channels[0],
            "coin_map": channels[1],
            "coins_in_quartal_description": ['OL', 'OR', 'UL', 'UR'],
            "coins_in_quartal": coins_in_quartal,
            "coin_indicator": max_coin_quartal
        }
    
    return np.append(np.stack(channels).reshape(-1), max_coin_quartal)

def state_to_features_old(game_state: dict, readable = False) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector. 
    :param game_state:  A dictionary describing the current game board.
    :return: np.array (NUM_FEATURES,)
    """
    
    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None
    
    pos = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']

    nearest_coin_path = get_nearest_coin_path(field, pos, coins)
    action_to_next_coin_features = np.array([0, 0, 0, 0])
   
    if not nearest_coin_path is None and len(nearest_coin_path) > 1:
        next_pos = nearest_coin_path[1]
        # oben
        action_to_next_coin_features[0] = 1 if (pos[0], pos[1]-1) == next_pos else 0
        # unten
        action_to_next_coin_features[1] = 1 if (pos[0], pos[1]+1) == next_pos else 0
        # rechts 
        action_to_next_coin_features[2] = 1 if (pos[0]+1, pos[1]) == next_pos else 0
        # linkes
        action_to_next_coin_features[3] = 1 if (pos[0]-1, pos[1]) == next_pos else 0
    
    
    return action_to_next_coin_features

def get_nearest_coin_path(field, pos, coins):
    """
    This function finds the path that need the fewest steps from the agents current position to the nearest coin
    :param field: a 2D numpy array of the field (empty, crates, walls)
    :param pos: a (x,y) tuple with the agents position
    :param coins: a 2D numpy array of the coin map
    :return: a list refers to the the path from the agent to the nearest coin
    """

    min_path = None
    min_path_val = 1000
    maze = shortest_path_map(field, pos)
    for coin in coins:
        path = shortest_path(field, pos, coin, maze)
        len_path = len(path)
        if len_path < min_path_val:
            min_path_val = len_path
            min_path = path

    return min_path


def shortest_path_map(field, pos):
    """
    This function finds the shortest path from the current position of the agent to any other point
    :param field: a 2D numpy array of the field (empty, crates, walls)
    :param pos: a (x,y) tuple with the agents position
    :return: a 2D numpy array of the maze, the nonzero entries describes
        the number of steps the agent needs to this position
    """

    # build the maze
    maze = np.zeros(field.shape)
    maze[pos] = 1

    def make_step(k):
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == k:
                    if i > 0 and maze[i - 1][j] == 0 and field[i - 1][j] == 0:
                        maze[i - 1][j] = k + 1
                    if j > 0 and maze[i][j - 1] == 0 and field[i][j - 1] == 0:
                        maze[i][j - 1] = k + 1
                    if i < len(maze) - 1 and maze[i + 1][j] == 0 and field[i + 1][j] == 0:
                        maze[i + 1][j] = k + 1
                    if j < len(maze[i]) - 1 and maze[i][j + 1] == 0 and field[i][j + 1] == 0:
                        maze[i][j + 1] = k + 1

    k = 0
    while True:
        k += 1
        make_step(k)
        if k>100: break

    return maze

def shortest_path(field, pos1, pos2, maze=None):
    """
    This function finds the shortest path between two positions on the map
    :param field: a 2D numpy array of the field (empty, crates, walls)
    :param pos1: a (x,y) tuple with the agents position
    :param pos2: a (x,y) tuple with the destination position
    :param maze: a 2D numpy array of the maze, the nonzero entries describes
        the number of steps the agent needs to this position, if not given, it has to be calculated
    :return: a list that contains the shortest path from our agents to the destination position
    """

    if maze is None:
        if field[pos2] == 1: field[pos2] = 0
        # build up the maze
        maze = np.zeros(field.shape)
        maze[pos1] = 1

        def make_step(k):
            for i in range(len(maze)):
                for j in range(len(maze[i])):
                    if maze[i][j] == k:
                        if i > 0 and maze[i - 1][j] == 0 and field[i - 1][j] == 0:
                            maze[i - 1][j] = k + 1
                        if j > 0 and maze[i][j - 1] == 0 and field[i][j - 1] == 0:
                            maze[i][j - 1] = k + 1
                        if i < len(maze) - 1 and maze[i + 1][j] == 0 and field[i + 1][j] == 0:
                            maze[i + 1][j] = k + 1
                        if j < len(maze[i]) - 1 and maze[i][j + 1] == 0 and field[i][j + 1] == 0:
                            maze[i][j + 1] = k + 1

        k = 0
        while maze[pos2[0]][pos2[1]] == 0:
            k += 1
            make_step(k)
            if k>100: return None


    i, j = pos2
    k = maze[i][j]
    path = [(i, j)]
    while k > 1:
        if i > 0 and maze[i - 1][j] == k - 1:
            i, j = i - 1, j
            path.append((i, j))
            k -= 1
        elif j > 0 and maze[i][j - 1] == k - 1:
            i, j = i, j - 1
            path.append((i, j))
            k -= 1
        elif i < len(maze) - 1 and maze[i + 1][j] == k - 1:
            i, j = i + 1, j
            path.append((i, j))
            k -= 1
        elif j < len(maze[i]) - 1 and maze[i][j + 1] == k - 1:
            i, j = i, j + 1
            path.append((i, j))
            k -= 1

    return path[::-1]
