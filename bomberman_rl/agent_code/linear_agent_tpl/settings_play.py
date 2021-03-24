# --- Policy settings
PLAY_POLICY_TYPE = 'SOFTMAX'

# Settings for PLAY_POLICY_TYPE == 'EPSILON-GREEDY'
EPSILON_PLAY = 0.2

# Settings for PLAY_POLICY_TYPE == 'SOFTMAX'
# lower -> more randomness; higher -> more greedy
INVERSE_TEMPERATURE_PLAY = 100
