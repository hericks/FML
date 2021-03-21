import os
import pickle
import random

import numpy as np

import logging
import sys


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']
NUM_ACTIONS = len(ACTIONS)

AGENT_NAME = "linear_agent_crate_first_try"

# 0 <= NUM_LOOK_AROUND <= 15
NUM_LOOK_AROUND = 3

EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 150]

EPSILON_PLAY = 0.35

STDOUT_LOGLEVEL = logging.DEBUG

# weights: np.array (NUM_ACTIONS, NUM_FEATURES)

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
        self.logger.info("Setting up model from scratch.")
        self.weights = np.zeros((NUM_ACTIONS, get_num_features()))
    else:
        self.logger.info("Loading model from saved state.")
        with open("weights.pt", "rb") as file:
            self.weights = pickle.load(file)
            
def evaluate_q(features, action, weights):
    return np.dot(weights, features)[ACTIONS.index(action)]
    
    
def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
  
    epsilon_train = np.interp(game_state['round'], EPSILON_TRAIN_BREAKS, EPSILON_TRAIN_VALUES)
    
    if (self.train and random.random() < epsilon_train) or (not self.train and random.random() < EPSILON_PLAY):
        # self.logger.debug("Choosing action purely at random.")
        return np.random.choice(ACTIONS, p=[.2, .2, .2, .2, .1, .1]) 
    
    action_map, _ = normalize_state(game_state)
    features = state_to_features(game_state)
    q_values = np.dot(self.weights, features)
    
    if not self.train:
        feature_dict = state_to_features(game_state, True)
        # self.logger.debug(f"Wall map: \n{feature_dict['wall_map']}")
        # self.logger.debug(f"Coin map: \n{feature_dict['coin_map']}")
        # self.logger.debug(f"Coins in quartal desc: {feature_dict['coins_in_quartal_description']}")
        # self.logger.debug(f"Coins in quartal: {feature_dict['coins_in_quartal']}")
        # self.logger.debug(f"Coin indicator: {feature_dict['coin_indicator']}")
        # self.logger.debug(f"Actions: {ACTIONS}")
        # self.logger.debug(f"Q-Values: {q_values}")
        # self.logger.debug(f"Choosing action {ACTIONS[np.argmax(q_values)]}.")
        # self.logger.debug(f"Real-Actions: {[action_map(a) for a in ACTIONS]}")
        # self.logger.debug(f"{game_state['step']}")
    
    return action_map(ACTIONS[np.argmax(q_values)])
    
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

    def action_map(a):
        if transposed_board:
            if a == 'RIGHT':
                a = 'DOWN'
            elif a == 'DOWN':
                a = 'RIGHT'
            elif a == 'LEFT':
                a = 'UP'
            elif a == 'UP':
                a = 'LEFT'
        if flipped_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flipped_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        return a
        
    def reverse_action_map(a):
        if flipped_x:
            a = 'RIGHT' if a == 'LEFT' else ('LEFT' if a == 'RIGHT' else a)
        if flipped_y:
            a = 'UP' if a == 'DOWN' else ('DOWN' if a == 'UP' else a)
        if transposed_board:
            if a == 'RIGHT':
                a = 'DOWN'
            elif a == 'DOWN':
                a = 'RIGHT'
            elif a == 'LEFT':
                a = 'UP'
            elif a == 'UP':
                a = 'LEFT'
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
    crate_map = np.zeros((31, 31))
    field = game_state['field'] 
    for x in np.arange(17):
        for y in np.arange(17):
            x_rel, y_rel = x - self_x, y - self_y
            if (field[x, y] == -1):
              wall_map[15 + y_rel, 15 + x_rel] = 1
            if (field[x, y] != 1):
              crate_map[15 + y_rel, 15 + x_rel] = 1
            
    coin_map = np.zeros((31, 31)) 
    coins = game_state['coins']
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[15 + y_rel, 15 + x_rel] = 1
   
    # stack maps     
    index_min = 15 - NUM_LOOK_AROUND
    index_max = 16 + NUM_LOOK_AROUND
    
    channels = []
    channels.append(wall_map[index_min:index_max,index_min:index_max])
    channels.append(coin_map[index_min:index_max,index_min:index_max])
    channels.append(crate_map[index_min:index_max,index_min:index_max])

    # extra features
    bombs = game_state['bombs']
    safe_death_features = get_safe_death_features((self_x, self_y), field, bombs)
    can_place_bomb = np.array([game_state['self'][2]], dtype = np.int32)
   
    features = np.stack(channels).reshape(-1)
    features = np.append(features, can_place_bomb)
    features = np.append(features, safe_death_features)
    return features

def get_unsafe_tiles(field, bombs):
  unsafe_positions = []
  for bomb_pos, _ in bombs:
    unsafe_positions.append(bomb_pos)
    
    for x_offset in range(1, 4):
      pos = (bomb_pos[0] + x_offset, bomb_pos[1])
      if pos[0] > 16 or field[pos] == -1:
        break
      unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)
      
    for x_offset in range(-1, -4, -1):
      pos = (bomb_pos[0] + x_offset, bomb_pos[1])
      if pos[0] < 0 or field[pos] == -1:
        break
      unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)
      
    for y_offset in range(1, 4):
      pos = (bomb_pos[0], bomb_pos[1] + y_offset)
      if pos[0] > 16 or field[pos] == -1:
        break
      unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)
      
    for y_offset in range(-1, -4, -1):
      pos = (bomb_pos[0], bomb_pos[1] + y_offset)
      if pos[0] < 0 or field[pos] == -1:
        break
      unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)
   
  return unsafe_positions

def get_reachable_tiles(pos, num_steps, field):
  if num_steps == 0:
    return [pos]
  elif num_steps == 1:
    ret = [pos]
    pos_x, pos_y = pos
 
    for pos_update in [(pos_x + 1, pos_y), (pos_x - 1, pos_y), (pos_x, pos_y + 1), (pos_x, pos_y - 1)]:
      if 0 <= pos_update[0] <= 16 and 0 <= pos_update[1] <= 16 and field[pos_update] == 0:
        ret.append(pos_update)
    
    return ret
  else:
    candidates = get_reachable_tiles(pos, num_steps - 1, field)
    ret = []
    for pos in candidates:
      ret.extend(x for x in get_reachable_tiles(pos, 1, field) if x not in ret)
    return ret
    
def get_reachable_safe_tiles(pos, field, bombs, look_ahead = True):
  if len(bombs) == 0:
    raise ValueError("No bombs placed.")
 
  timer =  bombs[0][1] if look_ahead else bombs[0][1] + 1
  reachable_tiles = set(get_reachable_tiles(pos, timer, field))
  unsafe_tiles = set(get_unsafe_tiles(field, bombs))
  
  return [pos for pos in reachable_tiles if pos not in unsafe_tiles] 
  
def is_safe_death(pos, field, bombs, look_ahead = True):
  if len(bombs) == 0:
    return False
    
  return len(get_reachable_safe_tiles(pos, field, bombs, look_ahead)) == 0
  
def get_safe_death_features(pos, field, bombs):
  if len(bombs) == 0:
    return np.array([0, 0, 0, 0, 0])
 
  ret = np.array([], dtype = np.int32) 
  for pos_update in [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1), (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), pos]:
    if field[pos_update] == 0:
      ret = np.append(ret, 1 if is_safe_death(pos_update, field, bombs) else 0)
    else:
      ret = np.append(ret, 1 if is_safe_death(pos, field, bombs) else 0)
  return ret
