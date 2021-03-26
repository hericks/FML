import numpy as np
import queue as Q
import heapq

def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')

    state_to_features(game_state)

    return game_state['user_input']


def state_to_features(game_state: dict, readable=False) -> np.array:
    """
    Converts the game state to the input of your model, i.e.
    a feature vector. 
    :param game_state:  A dictionary describing the current game board.
    :return: np.array (NUM_FEATURES,)
    """

    # This is the dict before the game begins and after it ends
    if game_state is None:
        return None

    self_x, self_y = game_state['self'][3]

    wall_map = np.zeros((31, 31))
    crate_map = np.zeros((31, 31))
    field = game_state['field']
    for x in np.arange(17):
        for y in np.arange(17):
            x_rel, y_rel = x - self_x, y - self_y
            if (field[x, y] == -1):
                wall_map[15 + y_rel, 15 + x_rel] = 1
            if (field[x, y] != 1):
                crate_map[15 + y_rel, 15 + x_rel] = 1

    coin_map = np.zeros((31, 31))
    coins = game_state['coins']
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[15 + y_rel, 15 + x_rel] = 1

    pos = game_state['self'][3]
    bombs = game_state['bombs']

    # print("\n\n\nCurrent state and now.") 
    # print(f"Reachable in 0 steps: {get_reachable_tiles(pos, 0, field)}")
    # print(f"Reachable in 1 steps: {get_reachable_tiles(pos, 1, field)}")
    # print(f"Reachable in 2 steps: {get_reachable_tiles(pos, 2, field)}")
    # 
    # if (len(bombs) == 0):
    #   print("No bombs placed. All tiles safe.")
    # else:
    #   print(f"Reachable safe tiles: {get_reachable_safe_tiles(pos, field, bombs, look_ahead = False)}")
    #   
    # print(f"Is save death: {is_safe_death(pos, field, bombs, look_ahead = False)}")
    # 
    # print(f"\n U D R L W")
    # print(f"{get_safe_death_features(pos, field, bombs)}")

    # print(f"{is_bomb_suicide(pos, field)}")

    free_tiles = np.zeros(field.shape, dtype=bool)
    for i in range(len(free_tiles)):
        for j in range(len(free_tiles[i])):
            if field[i, j] == 0:
                free_tiles[i, j] = 1

    print(f"Shortest path: {shortest_path(free_tiles, pos, (2, 3))}")


def get_unsafe_tiles(field, bombs):
    unsafe_positions = []
    for bomb_pos, _ in bombs:
        unsafe_positions.append(bomb_pos)

        for x_offset in range(1, 4):
            pos = (bomb_pos[0] + x_offset, bomb_pos[1])
            if pos[0] > 16 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for x_offset in range(-1, -4, -1):
            pos = (bomb_pos[0] + x_offset, bomb_pos[1])
            if pos[0] < 0 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for y_offset in range(1, 4):
            pos = (bomb_pos[0], bomb_pos[1] + y_offset)
            if pos[0] > 16 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

        for y_offset in range(-1, -4, -1):
            pos = (bomb_pos[0], bomb_pos[1] + y_offset)
            if pos[0] < 0 or field[pos] == -1:
                break
            unsafe_positions.extend(x for x in [pos] if x not in unsafe_positions)

    return unsafe_positions


def get_reachable_tiles(pos, num_steps, field):
    if num_steps == 0:
        return [pos]
    elif num_steps == 1:
        ret = [pos]
        pos_x, pos_y = pos

        for pos_update in [(pos_x + 1, pos_y), (pos_x - 1, pos_y), (pos_x, pos_y + 1), (pos_x, pos_y - 1)]:
            if 0 <= pos_update[0] <= 16 and 0 <= pos_update[1] <= 16 and field[pos_update] == 0:
                ret.append(pos_update)

        return ret
    else:
        candidates = get_reachable_tiles(pos, num_steps - 1, field)
        ret = []
        for pos in candidates:
            ret.extend(x for x in get_reachable_tiles(pos, 1, field) if x not in ret)
        return ret


def get_reachable_safe_tiles(pos, field, bombs, look_ahead=True):
    if len(bombs) == 0:
        raise ValueError("No bombs placed.")

    timer = bombs[0][1] if look_ahead else bombs[0][1] + 1
    reachable_tiles = set(get_reachable_tiles(pos, timer, field))
    unsafe_tiles = set(get_unsafe_tiles(field, bombs))

    return [pos for pos in reachable_tiles if pos not in unsafe_tiles]


def is_safe_death(pos, field, bombs, look_ahead=True):
    if len(bombs) == 0:
        return False

    return len(get_reachable_safe_tiles(pos, field, bombs, look_ahead)) == 0


def get_safe_death_features(pos, field, bombs):
    """

    :param pos:
    :param field:
    :param bombs:
    :return:
    """
    if len(bombs) == 0:
        return np.array([0, 0, 0, 0, 0])

    ret = np.array([], dtype=np.int32)
    for pos_update in [(pos[0], pos[1] - 1), (pos[0], pos[1] + 1), (pos[0] + 1, pos[1]), (pos[0] - 1, pos[1]), pos]:
        if field[pos_update] == 0:
            ret = np.append(ret, 1 if is_safe_death(pos_update, field, bombs) else 0)
        else:
            ret = np.append(ret, 1 if is_safe_death(pos, field, bombs) else 0)
    return ret


def is_bomb_suicide(pos, field):
    return is_safe_death(pos, field, [((pos), 3)], look_ahead=False)


class Node:
    """
    This class is needed to perform an A* search
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.g < other.g


def shortest_path(free_tiles, start, target):
    """
    This is an A* search algorithm to find the shortest path from start to target node
    :param free_tiles:
    :param start:
    :param target:
    :return:
    """

    start_node = Node(None, start)
    target_node = Node(None, target)
    start_node.g = start_node.h = start_node.f = 0
    target_node.g = target_node.h = target_node.f = 0

    open_nodes = []
    closed_nodes = []

    heapq.heappush(open_nodes, (start_node.g, start_node))

    while len(open_nodes) > 0:
        '''
        current_node = open_nodes[0]
        current_index = 0
        for index, item in enumerate(open_nodes):
            if item.f < current_node.f:
                current_node = item
                current_index = index
        '''

        current_node = heapq.heappop(open_nodes)[1]
        closed_nodes.append(current_node)

        if current_node == target_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        # get the current node and all its neighbors
        neighbors = []
        i, j = current_node.position
        neighbors_pos = [(i, j) for (i, j) in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)] if free_tiles[i, j]]

        for position in neighbors_pos:
            node_position = (position[0], position[1])
            new_node = Node(current_node, node_position)
            neighbors.append(new_node)

        for neighbor in neighbors:
            if neighbor in closed_nodes:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - target_node.position[0]) ** 2) + (
                    (neighbor.position[1] - target_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            if not any(node[1] == neighbor for node in open_nodes):
                heapq.heappush(open_nodes, (neighbor.g, neighbor))
                continue

            for open_node in open_nodes:
                if neighbor == open_node[1]:
                    open_node[1].g = min(neighbor.g, open_node[1].g)
                    break
