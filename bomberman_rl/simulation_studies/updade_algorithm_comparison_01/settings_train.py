import os
import events as e

# --- Simulation study
EXTERNAL_CONTROL = True

# Hyper parameters
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.75

# --- Policy settings
TRAIN_POLICY_TYPE = 'SOFTMAX'

# Settings for TRAIN_POLICY_TYPE == 'EPSILON-GREEDY'
EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 250]

# Settings for TRAIN_POLICY_TYPE == 'SOFTMAX'
# lower -> more randomness; higher -> more greedy
INVERSE_TEMPERATURE_TRAIN_VALUES = [75]
INVERSE_TEMPERATURE_TRAIN_BREAKS = [0]

# --- Learning settings
UPDATE_ALGORITHM = 'N-STEP-SARSA'

# Settings for UPDATE_ALGORITHM == 'SARSA'
None

# Settings for UPDATE_ALGORITHM == 'N-STEP-SARSA'
NUM_SARSA_STEPS = 1

# Settings for UPDATE_ALGORITHM == 'SARSA(LAMBDA)'
TRACE_DECAY = 0.9


# --- Reward settings
CONSTANT_REWARD = -0.1

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1
}

# --- History settings
AGENT_NAME = "N-STEP-SARSA-N=1"

# FOR EXTERNAL CONTROL (DO NOT CHANGE) -----------------------------------------

AGENT_NAME = os.environ['AGENT_NAME']
LEARNING_RATE = float(os.environ['LEARNING_RATE'])
UPDATE_ALGORITHM = os.environ['UPDATE_ALGORITHM']
NUM_SARSA_STEPS = int(os.environ['NUM_SARSA_STEPS'])
if 'TRACE_DECAY' in os.environ:
  TRACE_DECAY = float(os.environ['TRACE_DECAY'])
