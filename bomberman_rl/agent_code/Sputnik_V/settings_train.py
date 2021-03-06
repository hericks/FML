import os
import events as e

# --- Simulation study
EXTERNAL_CONTROL = True

# --- History settings
AGENT_NAME = "SARSA(LAMBDA)"

# Hyper parameters
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.75

# --- Policy settings
TRAIN_POLICY_TYPE = 'SOFTMAX'

# Settings for TRAIN_POLICY_TYPE == 'EPSILON-GREEDY'
EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 250]

# Settings for TRAIN_POLICY_TYPE == 'SOFTMAX'
# lower -> more randomness; higher -> more greedy
INVERSE_TEMPERATURE_TRAIN_VALUES = [75, 75, 150]
INVERSE_TEMPERATURE_TRAIN_BREAKS = [0, 300, 1000]

# --- Learning settings
UPDATE_ALGORITHM = 'SARSA(LAMBDA)'

# Settings for UPDATE_ALGORITHM == 'SARSA'
None

# Settings for UPDATE_ALGORITHM == 'N-STEP-SARSA'
NUM_SARSA_STEPS = 1

# Settings for UPDATE_ALGORITHM == 'SARSA(LAMBDA)'
TRACE_DECAY = 0.9

# --- Reward settings
CONSTANT_REWARD = -0.3

from .custom_event_utils import CRATE_DESTROYING_BOMB_DROPPED
from .custom_event_utils import BOMB_DROPPED_NO_CRATE_DESTROYED

EVENT_REWARDS = {
    e.COIN_COLLECTED: 32,
    e.KILLED_SELF: -38,
    e.CRATE_DESTROYED: 2,
    e.INVALID_ACTION: -1,
    CRATE_DESTROYING_BOMB_DROPPED: 5,
    BOMB_DROPPED_NO_CRATE_DESTROYED: -2,
    e.KILLED_OPPONENT: 75
}

# FOR EXTERNAL CONTROL (DO NOT CHANGE) -----------------------------------------

import simulation_study_keys as k


def valid_in_environ(key):
    return key in os.environ and os.environ[key] != 'None'


def assign_settings_from_environ():
    global AGENT_NAME
    if valid_in_environ(k.AGENT_NAME_KEY):
        AGENT_NAME = os.environ[k.AGENT_NAME_KEY]
    
    global LEARNING_RATE
    if valid_in_environ(k.LEARNING_RATE_KEY):
        LEARNING_RATE = float(os.environ[k.LEARNING_RATE_KEY])
    
    global DISCOUNT_FACTOR
    if valid_in_environ(k.DISCOUNT_FACTOR_KEY):
        DISCOUNT_FACTOR = float(os.environ[k.DISCOUNT_FACTOR_KEY])
    
    global UPDATE_ALGORITHM
    if valid_in_environ(k.UPDATE_ALGORITHM_KEY):
        UPDATE_ALGORITHM = os.environ[k.UPDATE_ALGORITHM_KEY]
        
    global NUM_SARSA_STEPS
    if valid_in_environ(k.NUM_SARSA_STEPS_KEY):
        NUM_SARSA_STEPS = int(os.environ[k.NUM_SARSA_STEPS_KEY])
    
    global TRACE_DECAY
    if valid_in_environ(k.TRACE_DECAY_KEY):
        TRACE_DECAY = float(os.environ[k.TRACE_DECAY_KEY])


if EXTERNAL_CONTROL:
    assign_settings_from_environ()
