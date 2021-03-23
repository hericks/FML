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
DISCOUNT_FACTOR = 0.75

# --- Policy settings
TRAIN_POLICY_TYPE = ['EPSILON_GREEDY', 'SOFTMAX']

# Settings for TRAIN_POLICY_TYPE == 'EPSILON_GREEDY'
EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 150]

# Settings for TRAIN_POLICY_TYPE == 'SOFTMAX'
TEMPERATURE_TRAIN_VALUES = [0.25, 0.1]
TEMPERATURE_TRAIN_BREAKS = [0, 150]

# --- Learning settings
UPDATE_ALGORITHM = 'N-STEP_SARSA'

# Settings for UPDATE_ALGORITHM == 'SARSA'
None

# Settings for UPDATE_ALGORITHM == 'N-STEP-SARSA'
NUM_SARSA_STEPS = 5

# --- History settings
AGENT_NAME = "linear_agent_tpl"

# Further objects
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    self.transition_history = []
  
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
    
    # update history (important: using raw states)
    update_history_from_transition(self, old_game_state, self_action, new_game_state, events)

    # normalize states; extract features; normalize action;
    _, reverse_action_map = normalize_state(old_game_state)
    _, _                  = normalize_state(new_game_state)
    self_action = reverse_action_map(self_action)
    
    # initialize transition object and append to transition history
    if not self_action is None: 
      transition = Transition(
          state_to_features(old_game_state),
          self_action,
          state_to_features(new_game_state),
          reward_from_events(self, events))
      self.transition_history.append(transition)
    
    # take update step; based exclusively on transition history
    perform_weight_update(self)

def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.

    This is similar to reward_update. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """

    # update history (important: using raw states)
    update_history_from_terminal(self, last_game_state, last_action, events)
    log_most_recent_history_entries(self)
    save_history(self)
    init_new_history_values(self)
    
    # normalize state and action
    _, reverse_action_map = normalize_state(last_game_state)
    last_action = reverse_action_map(last_action)
    
    # initialize transition object and append to transition history
    # note that we have to rely on the transition history for the previous state
    transition = Transition(
        self.transition_history[-1][3],
        last_action,
        state_to_features(last_game_state),
        reward_from_events(self, events)
    )
    self.transition_history.append(transition)
    
    # take update step; based exclusively on transition history
    perform_weight_update(self, terminal=True)
    
    # reset transition history
    self.transition_history = []
    
    # store the model
    with open("weights.pt", "wb") as file:
        pickle.dump(self.weights, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify rewards to en/discourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]

    return reward_sum - 0.01

# ------------------------------------------------------------------------------
# learning algorithms ----------------------------------------------------------
# ------------------------------------------------------------------------------

def perform_weight_update(self, terminal=False):
  if UPDATE_ALGORITHM == 'SARSA':
      perform_sarsa_weight_update(self, terminal)
  elif UPDATE_ALGORITHM == 'N-STEP-SARSA':
      if not terminal:
          perform_n_step_sarsa_weight_update(self, NUM_SARSA_STEPS)
      else:
          perform_terminal_n_step_sarsa_weight_update(self, NUM_SARSA_STEPS)
  else:
      raise NotImplementedError(f"Update algorithm '{UPDATE_ALGORITHM}' is not implemented yet.")
      
def perform_n_step_sarsa_weight_update(self, n):
    
  
  
def perform_terminal_n_step_sarsa_weight_update(self, n):
  
  
  
  
  
  
  
 
 
 
  
  
      
def perform_sarsa_weight_update(self, terminal):
    if len(self.transition_history) == 0:
        return
      
    S, A, S_new, R_new = self.transition_history[-1]
  
    if len(self.transition_history) > 2:
        S_old, A_old, _, R = self.transition_history[-2]
        diff = R + DISCOUNT_FACTOR * evaluate_q(S, A, self.weights) - evaluate_q(S_old, A_old, self.weights)
        derivative = np.zeros_like(self.weights)
        derivative[ACTIONS.index(A_old)] = S_old
        self.weights = self.weights + LEARNING_RATE * diff * derivative
      
    if terminal:
        diff = R_new - evaluate_q(S, A, self.weights)
        derivative = np.zeros_like(self.weights)
        derivative[ACTIONS.index(A)] = S
        self.weights = self.weights + LEARNING_RATE * diff * derivative

# Further objects
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    
def perform_semi_gradient_sarsa_update(self, S, A, R, S_new, A_new):
    weights = self.weights 
    diff = R - evaluate_q(S, A, weights)
    if not S_new is None:
        diff += DISCOUNT_FACTOR * evaluate_q(S_new, A_new, weights)
    derivative = np.zeros_like(weights)
    derivative[ACTIONS.index(A)] = S
    self.weights = weights + LEARNING_RATE * diff * derivative

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
    self.logger.info(f"{key}: {np.round(self.history[key][-1], 2)}")
  print("\n")
  
def save_history(self):
  with open(self.history_filepath, "wb") as file:
    pickle.dump(self.history, file)
