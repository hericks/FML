import os
import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, evaluate_q, ACTIONS, AGENT_NAME, normalize_state
from .callbacks import EPSILON_TRAIN_VALUES, EPSILON_TRAIN_BREAKS

import matplotlib.pyplot as plt
from datetime import datetime

# Hyper parameters
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.995

TRAIN_POLICY_TYPE = ['EPSILON_GREEDY', 'SOFTMAX']

# Greedy settings
EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 150]

# Softmax settings
TEMPERATURE_TRAIN_VALUES = [0.25, 0.1]
TEMPERATURE_TRAIN_BREAKS = [0, 150]

# History settings
AGENT_NAME = "linear_agent_tpl"

# Further objects
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    self.lastTransition = None
  
    # setup everything related to history / monitoring
    setup_history(self)
    init_new_history_values(self)

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """

    _, reverse_action_map = normalize_state(old_game_state)
    normalize_state(new_game_state)

    S = state_to_features(old_game_state)
    A = reverse_action_map(self_action)
    R_new = reward_from_events(self, events)
    S_new = state_to_features(new_game_state)
    
    if self.lastTransition != None:
        S_old = self.lastTransition.state
        A_old = self.lastTransition.action
        R = self.lastTransition.reward
        
        if A_old != None:
            perform_semi_gradient_sarsa_update(self, S_old, A_old, R, S, A)

    self.lastTransition = Transition(S, A, S_new, R_new)

    # update history
    update_history_from_transition(self, old_game_state, self_action, new_game_state, events)
    
def perform_semi_gradient_sarsa_update(self, S, A, R, S_new, A_new):
    weights = self.weights 
    diff = R - evaluate_q(S, A, weights)
    if not S_new is None:
      diff += DISCOUNT_FACTOR * evaluate_q(S_new, A_new, weights)
    derivative = np.zeros_like(weights)
    derivative[ACTIONS.index(A)] = S
    self.weights = weights + LEARNING_RATE * diff * derivative

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    _, reverse_action_map = normalize_state(last_game_state)
    S = state_to_features(last_game_state)
    A = reverse_action_map(last_action)
    R = reward_from_events(self, events)
    perform_semi_gradient_sarsa_update(self, S, A, R, None, None)
      
    # store model history
    update_history_from_terminal(self, last_game_state, last_action, events)
    log_most_recent_history_entries(self)
    save_history(self)
    init_new_history_values(self)
    
    # store the model
    with open("weights.pt", "wb") as file:
      pickle.dump(self.weights, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify rewards to en/discourage certain behavior.
    """
    game_rewards = {
        e.BOMB_DROPPED: 1,
        e.KILLED_SELF: -50,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum

# ------------------------------------------------------------------------------
# history ----------------------------------------------------------------------
# ------------------------------------------------------------------------------

def setup_history(self):
  self.history_folder = f"histories/{AGENT_NAME}"
  self.history_filepath = datetime.now().strftime(f"{self.history_folder}/%d_%m_%Y_%H_%M_%S_%f.pt")
 
  self.logger.info(f"Creating history directory {self.history_filepath}...") 
  if not os.path.exists(self.history_folder):
    try:
      os.mkdir(self.history_folder)
    except OSError:
      self.logger.info(f"{self.history_folder} creation failed.")
    else:
      self.logger.info(f"{self.history_folder} creation successful.")
  else:
      self.logger.info(f"{self.history_folder} already exists. Nothing to do.")
  
  history_keys = [
    'cumulative_reward',
    'num_bombs_dropped',
    'num_coins_collected',
    'num_crates_destroyed',
    'num_invalid_actions',
    'round_length'
  ]    
  
  self.history = dict()
  
  for key in history_keys:
    self.history[key] = []

def init_new_history_values(self):
  for key in self.history.keys():
    self.history[key].append(0)

def update_history_from_transition(self, old_game_state, self_action, new_game_state, events):
  # update cumulative reward 
  self.history['cumulative_reward'][-1] += reward_from_events(self, events)
 
  # update event counts 
  for event in events:
    if event == e.BOMB_DROPPED:
      self.history['num_bombs_dropped'][-1] += 1
    elif event == e.COIN_COLLECTED:
      self.history['num_coins_collected'][-1] += 1
    elif event == e.CRATE_DESTROYED:
      self.history['num_crates_destroyed'][-1] += 1
    elif event == e.INVALID_ACTION:
      self.history['num_invalid_actions'][-1] += 1
  
  self.history['round_length'][-1] = new_game_state['step']

def update_history_from_terminal(self, last_game_state, last_action, events):
  # default: update like transition
  update_history_from_transition(self, None, last_action, last_game_state, events)
  
def log_most_recent_history_entries(self):
  for key in self.history.keys():
    self.logger.info(f"{key}: {self.history[key][-1]}")
  print("\n")
  
def save_history(self):
  with open(self.history_filepath, "wb") as file:
    pickle.dump(self.history, file)
