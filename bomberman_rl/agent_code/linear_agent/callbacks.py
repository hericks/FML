import os
import pickle
import random

import numpy as np


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']

NUM_ACTIONS = len(ACTIONS)
NUM_FEATURES = 2*29*29

EPSILON = 1/3

# weights: np.array (NUM_ACTIONS, NUM_FEATURES)

def setup(self):
    """
    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train or not os.path.isfile("my-saved-model.pt"):
        self.logger.info("Setting up model from scratch.")
        self.weights = np.random.rand(NUM_ACTIONS, NUM_FEATURES)
    else:
        self.logger.info("Loading model from saved state.")
        with open("my-saved-model.pt", "rb") as file:
            self.weights = pickle.load(file)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    
    if self.train and random.random() < EPSILON:
        self.logger.debug("Choosing action purely at random.") # 80%: walk in any direction. 10% wait. 10% bomb.
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1])
        
    
    features = state_to_features(game_state)
    q_values = np.dot(self.weights, features)
    return ACTIONS[np.argmax(q_values)]


def state_to_features(game_state: dict) -> np.array:
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
   
    wall_map = np.zeros((29, 29)) 
    field = game_state['field'] 
    for x in np.arange(1, 16):
        for y in np.arange(1, 16):
            if (field[x, y] != -1):
                continue 
            
            x_rel, y_rel = x - self_x, y - self_y
            wall_map[14 + x_rel, 14 + y_rel] = 1
            
    coin_map = np.zeros((29, 29)) 
    coins = game_state['coins']
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[14 + x_rel, 14 + y_rel] = 1
        
    channels = [wall_map, coin_map]

    return np.stack(channels).reshape(-1)
