import events as e

# Hyper parameters
LEARNING_RATE = 0.0001
DISCOUNT_FACTOR = 0.75

# --- Policy settings
TRAIN_POLICY_TYPE = 'SOFTMAX'

# Settings for TRAIN_POLICY_TYPE == 'EPSILON-GREEDY'
EPSILON_TRAIN_VALUES = [0.25, 0.1]
EPSILON_TRAIN_BREAKS = [0, 150]

# Settings for TRAIN_POLICY_TYPE == 'SOFTMAX'
# lower -> more greedy; higher -> more randomness
INVERSE_TEMPERATURE_TRAIN_VALUES = [1000, 100]
INVERSE_TEMPERATURE_TRAIN_BREAKS = [0, 150]

# --- Learning settings
UPDATE_ALGORITHM = 'N-STEP-SARSA'

# Settings for UPDATE_ALGORITHM == 'SARSA'
None

# Settings for UPDATE_ALGORITHM == 'N-STEP-SARSA'
NUM_SARSA_STEPS = 5

# --- Reward settings
CONSTANT_REWARD = -0.1

EVENT_REWARDS = {
    e.COIN_COLLECTED: 1,
}

# --- History settings
AGENT_NAME = "linear_agent_tpl"
