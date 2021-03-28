def get_reachable_safe_tiles(pos, field, bombs, look_ahead=True):
    if len(bombs) == 0:
        raise ValueError("No bombs placed.")

    timer = bombs[0][1] if look_ahead else bombs[0][1] + 1
    reachable_tiles = set(get_reachable_tiles(pos, timer, field))
    unsafe_tiles = set(get_unsafe_tiles(field, bombs))

    return [pos for pos in reachable_tiles if pos not in unsafe_tiles]
