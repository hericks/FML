import pickle
import random

import numpy as np
import heapq

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
    if STDOUT_LOGLEVEL is not None:
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
    return np.random.choice(len(q_values), p=prob)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board. :return: The action to take as a string.
    """
    action_map, _ = normalize_state(game_state)
    features = state_to_features(game_state)
    q_values = np.dot(self.weights, features)

    if self.train:
        if TRAIN_POLICY_TYPE == 'EPSILON-GREEDY':
            epsilon_train = np.interp(game_state['round'], EPSILON_TRAIN_BREAKS, EPSILON_TRAIN_VALUES)
            action_index = epsilon_greedy(q_values, epsilon_train)
        elif TRAIN_POLICY_TYPE == 'SOFTMAX':
            temperature_train = np.interp(game_state['round'], INVERSE_TEMPERATURE_TRAIN_BREAKS,
                                          1 / np.array(INVERSE_TEMPERATURE_TRAIN_VALUES))
            action_index = softmax(q_values, temperature_train)
        else:
            raise NotImplementedError(f"The policy type '{TRAIN_POLICY_TYPE}' is not implemented.")
    else:
        if PLAY_POLICY_TYPE == 'EPSILON-GREEDY':
            action_index = epsilon_greedy(q_values, EPSILON_PLAY)
        elif PLAY_POLICY_TYPE == 'SOFTMAX':
            action_index = softmax(q_values, 1 / INVERSE_TEMPERATURE_PLAY)
        else:
            raise NotImplementedError(f"The policy type '{PLAY_POLICY_TYPE}' is not implemented.")

    return action_map(ACTIONS[action_index])


def normalize_state(game_state):
    """
    :param game_state: The dictionary that to normalize (in-place).

    :return: action_map: function to map action in normalized state to action in input_state,
    reverse_action_map: function to map action in input_state to action in normalized state.

    """
    if game_state is None:
        return lambda a: a, lambda a: a

    agent_x, agent_y = game_state['self'][3]

    def flip_tuple_x(t):
        return 16 - t[0], t[1]

    def flip_tuple_y(t):
        return t[0], 16 - t[1]

    flipped_x = False
    if agent_x > 8:
        game_state['field'] = np.flipud(game_state['field'])
        game_state['bombs'] = [(flip_tuple_x(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.flipud(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_x(coin) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, flip_tuple_x(pos))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple_x(pos)) for name, score, can_place_bomb, pos in
                                game_state['others']]
        flipped_x = True

    flipped_y = False
    if agent_y > 8:
        game_state['field'] = np.fliplr(game_state['field'])
        game_state['bombs'] = [(flip_tuple_y(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.fliplr(game_state['explosion_map'])
        game_state['coins'] = [flip_tuple_y(coin) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, flip_tuple_y(pos))
        game_state['others'] = [(name, score, can_place_bomb, flip_tuple_y(pos)) for name, score, can_place_bomb, pos in
                                game_state['others']]
        flipped_y = True

    agent_x_update, agent_y_update = game_state['self'][3]

    def transpose_tuple(t):
        return t[1], t[0]

    transposed_board = False
    if agent_y_update > agent_x_update:
        game_state['field'] = np.transpose(game_state['field'])
        game_state['bombs'] = [(transpose_tuple(pos), time) for pos, time in game_state['bombs']]
        game_state['explosion_map'] = np.transpose(game_state['explosion_map'])
        game_state['coins'] = [transpose_tuple(coin) for coin in game_state['coins']]
        name, score, can_place_bomb, pos = game_state['self']
        game_state['self'] = (name, score, can_place_bomb, transpose_tuple(pos))
        game_state['others'] = [(name, score, can_place_bomb, transpose_tuple(pos)) for name, score, can_place_bomb, pos
                                in game_state['others']]
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

from .feature_utils import *

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

    pos = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']

    return get_first_step_to_nearest_object_features(get_free_tiles(field), pos, coins, 2)
