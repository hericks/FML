def append_custom_events(game_state, action, events):
    # CRATE_DESTROYING_BOMB_DROPPED
    if action == 'BOMB' and will_bomb_destroy_crates(game_state['field'], game_state['self'][3]):
        events.append(CRATE_DESTROYING_BOMB_DROPPED)

    # BOMB_DROPPED_NO_CRATE_DESTROYED
    if action == 'BOMB' and not will_bomb_destroy_crates(game_state['field'], game_state['self'][3]):
        events.append(BOMB_DROPPED_NO_CRATE_DESTROYED)
