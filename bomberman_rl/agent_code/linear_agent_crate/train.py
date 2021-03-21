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

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.7


def setup_training(self):
    """
    Initialise self for training purpose.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.lastTransition = None
    
    self.history = dict()
    self.history['numCoinsCollected'] = deque()
    self.history['numInvalidActions'] = deque()
    self.history['roundLength'] = deque()
    self.history['epsilon'] = deque()
    self.history['numCratesDestroyed'] = deque()
   
    self.historyFolder = f"histories/{AGENT_NAME}"
    self.historyFilePath = datetime.now().strftime(f"{self.historyFolder}/%d_%m_%Y_%H_%M_%S_%f.pt")
    
    self.numInvalidActions = 0
    self.numCoinsCollected = 0
    self.numCratesDestroyed = 0
    
    if not os.path.exists(self.historyFolder):
      try:
        os.mkdir(self.historyFolder)
      except OSError:
        self.logger.info(f"Creation of the directory {self.historyFolder} failed")
      else:
        self.logger.info(f"Successfully created the directory {self.historyFolder}")
    else:
        self.logger.info(f"The directory {self.historyFolder} already exists.")
        
    
    self.logger.info(f"Saving history to {self.historyFilePath}.")

def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    """
    Called once per step to allow intermediate rewards based on game events.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    :param old_game_state: The state that was passed to the last call of `act`.
    :param self_action: The action that you took.
    :param new_game_state: The state the agent is in now.
    :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
    """
    # Idea: Add your own events to hand out rewards
    # if ...:
    #     events.append(PLACEHOLDER_EVENT)

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
    
    # logging
    if e.INVALID_ACTION in events:
      self.numInvalidActions += 1
        
    if e.COIN_COLLECTED in events:
      self.numCoinsCollected += 1
     
    for event in events:
      if event == e.CRATE_DESTROYED:
        self.numCratesDestroyed += 1
    
def perform_semi_gradient_sarsa_update(self, S, A, R, S_new, A_new):
    weights = self.weights 
    diff = R - evaluate_q(S, A, weights)
    if not S_new is None:
      diff += DISCOUNT_FACTOR * evaluate_q(S_new, A_new, weights)
    derivative = np.zeros_like(weights)
    derivative[ACTIONS.index(A)] = S
    # print(diff)
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
    
    # logging
    if e.INVALID_ACTION in events:
      self.numInvalidActions += 1
        
    if e.COIN_COLLECTED in events:
      self.numCoinsCollected += 1
     
    for event in events:
      if event == e.CRATE_DESTROYED:
        self.numCratesDestroyed += 1
    
    _, reverse_action_map = normalize_state(last_game_state)
    S = state_to_features(last_game_state)
    A = reverse_action_map(last_action)
    R = reward_from_events(self, events)
    perform_semi_gradient_sarsa_update(self, S, A, R, None, None)

    # update history
    self.history['numInvalidActions'].append(self.numInvalidActions)
    self.history['numCoinsCollected'].append(self.numCoinsCollected)
    self.history['roundLength'].append(last_game_state['step'])
    self.history['epsilon'].append(np.interp(last_game_state['round'], EPSILON_TRAIN_BREAKS, EPSILON_TRAIN_VALUES))
    self.history['numCratesDestroyed'].append(self.numCratesDestroyed)
    
    # logging 
    self.logger.info(f'{self.numInvalidActions} invalid moves were played.')
    self.logger.info(f'{self.numCoinsCollected} coins were collected.')
    self.logger.info(f'{self.numCratesDestroyed} crates were destroyed.')
    self.logger.info(f'The game went for {last_game_state["step"]} steps.')
    
    normalize_state(last_game_state)
    self.logger.info(np.dot(self.weights, state_to_features(last_game_state)))
    
    # store the model
    with open("weights.pt", "wb") as file:
      pickle.dump(self.weights, file)
      
    # store model history
    with open(self.historyFilePath, "wb") as file:
      pickle.dump(self.history, file)
      
    # perform reset 
    self.lastTransition = None
    self.numInvalidActions = 0
    self.numCoinsCollected = 0
    self.numCratesDestroyed = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify rewards to en/discourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 5,
        e.CRATE_DESTROYED: 1,
        e.KILLED_SELF: -10
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum - 0.01
