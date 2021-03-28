def get_num_features():
    dummy_state = {
        'round': 0,
        'step': 0,
        'field': np.zeros((17, 17)),
        'bombs': [],
        'explosion_map': np.zeros((17, 17)),
        'coins': [],
        'self': ("dummy", 0, True, (1, 1)),
        'others': []
    }

    return state_to_features(dummy_state).shape[0]