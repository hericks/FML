import os
import simulation_study_keys as k

NUM_TRAINING_SESSIONS = 10
N_ROUNDS = 75

LEARNING_RATES = [0.0002, 0.0003, 0.0004]
DISCOUNT_FACTORS = [None]

UPDATE_ALGORITHMS = ['N-STEP-SARSA', 'SARSA(LAMBDA)']

N_SARSA_STEPS = [1, 3, 5, 7]

TRACE_DECAYS = [0.75, 0.85, 0.95]


def inner_loop():
    for iteration in range(NUM_TRAINING_SESSIONS):
        print(f"Training session {iteration + 1}/{NUM_TRAINING_SESSIONS} of {os.environ[k.AGENT_NAME_KEY]}.")
        os.system(f"python main.py play --agents linear_agent_tpl --train 1 --n-rounds {N_ROUNDS} --no-gui")


def build_agent_name():
    ret = update_algorithm

    if update_algorithm == 'N-STEP-SARSA':
        ret += "" if num_sarsa_steps is None else f"-N={num_sarsa_steps}"
    elif update_algorithm == 'SARSA(LAMBDA)':
        ret += "" if trace_decay is None else f"-TRACE-DECAY={trace_decay:.3f}"

    ret += "" if discount_factor is None else f"-DISCOUNT-FACTOR={discount_factor:.3f}"
    ret += "" if alpha is None else f"-LEARNING-RATE={alpha:.6f}"

    return ret


for alpha in LEARNING_RATES:
    os.environ[k.LEARNING_RATE_KEY] = str(alpha)
    for discount_factor in DISCOUNT_FACTORS:
        os.environ[k.DISCOUNT_FACTOR_KEY] = str(discount_factor)
        for update_algorithm in UPDATE_ALGORITHMS:
            os.environ[k.UPDATE_ALGORITHM_KEY] = update_algorithm
            if update_algorithm == 'N-STEP-SARSA':
                for num_sarsa_steps in N_SARSA_STEPS:
                    os.environ[k.NUM_SARSA_STEPS_KEY] = str(num_sarsa_steps)
                    os.environ[k.AGENT_NAME_KEY] = build_agent_name()
                    inner_loop()
            elif update_algorithm == 'SARSA(LAMBDA)':
                for trace_decay in TRACE_DECAYS:
                    os.environ[k.TRACE_DECAY_KEY] = str(trace_decay)
                    os.environ[k.AGENT_NAME_KEY] = build_agent_name()
                    inner_loop()
            else:
                raise ValueError(f"Unknown update algorithm: {update_algorithm}")
