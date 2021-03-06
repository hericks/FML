# --- Definitions of custom events
CRATE_DESTROYING_BOMB_DROPPED = 'CRATE_DESTROYING_BOMB_DROPPED'
CRATE_DESTROYING_BOMB_DROPPED_WITHOUT_DYING = 'CRATE_DESTROYING_BOMB_DROPPED_WITHOUT_DYING'
BOMB_DROPPED_NO_CRATE_DESTROYED = 'BOMB_DROPPED_NO_CRATE_DESTROYED'


# --- Helper functions to compute these events
def is_pos_on_field(pos):
    return 0 <= pos[0] <= 16 and 0 <= pos[1] <= 16


def get_positions_reached_by_bomb(field, bomb_pos):
    reached_positions = [bomb_pos]

    offset_ranges = [range(1, 4), range(-1, -4, -1)]
    for offset_range in offset_ranges:
        for x_offset in offset_range:
            new_pos = (bomb_pos[0] + x_offset, bomb_pos[1])
            if not is_pos_on_field(new_pos) or field[new_pos] == -1:
                break
            else:
                reached_positions.append(new_pos)
        for y_offset in offset_range:
            new_pos = (bomb_pos[0], bomb_pos[1] + y_offset)
            if not is_pos_on_field(new_pos) or field[new_pos] == -1:
                break
            else:
                reached_positions.append(new_pos)

    return reached_positions


def num_crates_bomb_will_destroy(field, bomb_pos):
    positions_reached_by_bomb = get_positions_reached_by_bomb(field, bomb_pos)
    return len([pos for pos in positions_reached_by_bomb if field[pos] == 1])


def will_bomb_destroy_crates(field, bomb_pos):
    return num_crates_bomb_will_destroy(field, bomb_pos) > 0
