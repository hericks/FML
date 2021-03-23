import os
import pickle
import random
import numpy as np
from collections import namedtuple, deque
from datetime import datetime
from typing import List

from .callbacks import state_to_features, evaluate_q, ACTIONS, AGENT_NAME, normalize_state
from .settings_train import *

# Further objects
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def setup_training(self):
    self.transition_history = []
    self.transition_history_complete = False
  
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
    self.transition_history_complete = True
    
    # take update step; based exclusively on transition history
    perform_weight_update(self, terminal=True)
    
    # reset transition history
    self.transition_history = []
    self.transition_history_complete = False
    
    # store the model
    with open("weights.pt", "wb") as file:
        pickle.dump(self.weights, file)

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify rewards to en/discourage certain behavior.
    """
    reward_sum = 0
    for event in events:
        if event in EVENT_REWARDS:
            reward_sum += EVENT_REWARDS[event]

    return reward_sum + CONSTANT_REWARD

# ------------------------------------------------------------------------------
# learning algorithms ----------------------------------------------------------
# ------------------------------------------------------------------------------

def perform_weight_update(self, terminal=False):
    if UPDATE_ALGORITHM == 'SARSA':
        if not terminal:
            tau = len(self.transition_history) - 2
            perform_n_step_sarsa_weight_update(self, tau, 1)
        else:
            for tau in np.arange(len(self.transition_history) - 2, len(self.transition_history)):
                perform_n_step_sarsa_weight_update(self, tau, 1)
    elif UPDATE_ALGORITHM == 'N-STEP-SARSA':
        if not terminal:
            tau = len(self.transition_history) - NUM_SARSA_STEPS - 1
            perform_n_step_sarsa_weight_update(self, tau, NUM_SARSA_STEPS)
        else:
            for tau in np.arange(len(self.transition_history) - NUM_SARSA_STEPS - 1, len(self.transition_history)):
                perform_n_step_sarsa_weight_update(self, tau, NUM_SARSA_STEPS)
    else:
        raise NotImplementedError(f"Update algorithm '{UPDATE_ALGORITHM}' is not implemented yet.")
      
def perform_n_step_sarsa_weight_update(self, tau, n):
    if tau < 0:
        return
  
    # get S_tau, A_tau 
    S_tau, A_tau, _, _ = self.transition_history[tau]
    
    # compute initial q_tau
    q_tau = evaluate_q(S_tau, A_tau, self.weights)
    
    # computer gradient of initial q_tau
    gradient_q_tau = np.zeros_like(self.weights)
    gradient_q_tau[ACTIONS.index(A_tau)] = S_tau
    
    # compute update target with true rewards
    G = 0
    for i in np.arange(n):
        if tau + i >= len(self.transition_history):
            break
        G += np.power(DISCOUNT_FACTOR, i) * self.transition_history[tau+i][3]
    
    # add current reward estimate to the update target
    if tau + n < len(self.transition_history):
        S_tau_plus_n, A_tau_plus_n, _, _ = self.transition_history[tau+n]
        G += np.power(DISCOUNT_FACTOR, n) * evaluate_q(S_tau_plus_n, A_tau_plus_n, self.weights)
        
    self.weights = self.weights + LEARNING_RATE * (G - q_tau) * gradient_q_tau

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
