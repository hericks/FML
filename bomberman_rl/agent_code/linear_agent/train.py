import pickle
import random
import numpy as np
from collections import namedtuple, deque
from typing import List

import events as e
from .callbacks import state_to_features, evaluate_q, ACTIONS

import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# Hyper parameters
LEARNING_RATE = 0.000125
DISCOUNT_FACTOR = 0.999


def setup_training(self):
    """
    Initialise self for training purpose.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.lastTransition = None
    self.numInvalidActions = 0
    self.numInvalidActionsHistory = deque()
    self.numCoinsCollected = 0
    self.numCoinsCollectedHistory = deque()

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

    S = state_to_features(old_game_state)
    A = self_action
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
    
    
def perform_semi_gradient_sarsa_update(self, S, A, R, S_new, A_new):
    weights = self.weights 
    diff = R + DISCOUNT_FACTOR * evaluate_q(S_new, A_new, weights) - evaluate_q(S, A, weights)
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
    # self.logger.debug(f'Encountered event(s) {", ".join(map(repr, events))} in final step')
    # self.transitions.append(Transition(state_to_features(last_game_state), last_action, None, reward_from_events(self, events)))

    # update history
    self.numInvalidActionsHistory.append(self.numInvalidActions)
    self.numCoinsCollectedHistory.append(self.numCoinsCollected)
    
    # logging 
    self.logger.info(f'{self.numInvalidActions} invalid moves were played during this game.')
    self.logger.info(f'{self.numCoinsCollected} coins were collected during this game.')
    self.logger.info(f'The game went for {last_game_state["step"]} steps.')
    
    # store the model
    with open("weights.pt", "wb") as file:
      pickle.dump(self.weights, file)
      
    # store model history
    with open("history.pt", "wb") as file:
      pickle.dump((self.numInvalidActionsHistory, self.numCoinsCollectedHistory), file)
      
    # perform reset 
    self.lastTransition = None
    self.numInvalidActions = 0
    self.numCoinsCollected = 0

def reward_from_events(self, events: List[str]) -> int:
    """
    Modify rewards to en/discourage certain behavior.
    """
    game_rewards = {
        e.COIN_COLLECTED: 1,
        e.KILLED_OPPONENT: 5,
        e.INVALID_ACTION: -1,
    }
    reward_sum = 0
    for event in events:
        if event in game_rewards:
            reward_sum += game_rewards[event]
    # self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
    return reward_sum
