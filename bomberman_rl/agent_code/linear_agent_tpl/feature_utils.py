"""
--------------------------------------------------------------------------------
functions for state_to_features ------------------------------------------------
--------------------------------------------------------------------------------
"""


# This function is needed to get all free tiles of the field
# This is required for example for shortest path features
def get_free_tiles(field):
    """
    This function takes the field and returns all tiles where the agent could walk
    :param field: a 2D numpy array of the field
    :return: a 2D numpy array of type bool
    """

    free_tiles = np.zeros(field.shape, dtype=bool)
    for i in range(len(free_tiles)):
        for j in range(len(free_tiles[i])):
            if field[i, j] == 0:
                free_tiles[i, j] = 1

    return free_tiles


def get_coin_features(free_tiles, pos, coins, offset):
    """

    :param free_tiles:
    :param pos:
    :param coins:
    :param offset:
    :return:
    """

    nearest_coin_path = get_nearest_coin_path(free_tiles, pos, coins, offset)
    action_to_next_coin_features = np.array([0, 0, 0, 0])
    if nearest_coin_path is not None and len(nearest_coin_path) > 1:
        next_pos = nearest_coin_path[1]
        # UP
        action_to_next_coin_features[0] = 1 if (pos[0], pos[1] - 1) == next_pos else 0
        # DOWN
        action_to_next_coin_features[1] = 1 if (pos[0], pos[1] + 1) == next_pos else 0
        # RIGHT
        action_to_next_coin_features[2] = 1 if (pos[0] + 1, pos[1]) == next_pos else 0
        # LEFT
        action_to_next_coin_features[3] = 1 if (pos[0] - 1, pos[1]) == next_pos else 0

    return action_to_next_coin_features


def get_crate_features(field, free_tiles, pos, offset):
    """

    :param field:
    :param free_tiles:
    :param pos:
    :param offset:
    :return:
    """

    crates = [(x, y) for x in range(17) for y in range(17) if field[x, y] == 1]
    nearest_crate_path = get_nearest_coin_path(free_tiles, pos, crates, offset)
    action_to_next_crate_features = np.array([0, 0, 0, 0])

    if nearest_crate_path is not None and len(nearest_crate_path) > 1:
        next_pos = nearest_crate_path[1]
        # UP
        action_to_next_crate_features[0] = 1 if (pos[0], pos[1] - 1) == next_pos else 0
        # DOWN
        action_to_next_crate_features[1] = 1 if (pos[0], pos[1] + 1) == next_pos else 0
        # RIGHT
        action_to_next_crate_features[2] = 1 if (pos[0] + 1, pos[1]) == next_pos else 0
        # LEFT
        action_to_next_crate_features[3] = 1 if (pos[0] - 1, pos[1]) == next_pos else 0

    self_x, self_y = pos
    neighbors = [(self_x + 1, self_y), (self_x - 1, self_y), (self_x, self_y + 1), (self_x, self_y - 1)]
    is_next_to_crate = np.array(any([field[neighbor] == 1 for neighbor in neighbors]), dtype=np.int32)
    action_to_next_crate_features = np.append(action_to_next_crate_features, is_next_to_crate)

    return action_to_next_crate_features


"""
--------------------------------------------------------------------------------
relative map features ----------------------------------------------------------
--------------------------------------------------------------------------------
"""


# This function is needed to create relative maps from the agents position
def get_relative_maps(game_state):
    """
    This function takes the game_state and creates relative maps. The center (15, 15) is the agent's position
    :param game_state: a python dictionary that contains information about
        - the agent's position
        - the field (crates, walls, free tiles)
        - the position of the bombs and when the explode
        - a list of all coins
    :return: a dictionary with all the relative maps that will be needed
        'wall_map'
        'crate_map'
        'coin_map'
        'bomb_map_0'
        'bomb_map_1'
        'bomb_map_2'
        'bomb_map_3'
        'bomb_map_4'
    """

    # get attributes from game state
    pos = game_state['self'][3]
    field = game_state['field']
    coins = game_state['coins']
    bombs = game_state['bombs']
    self_x, self_y = pos

    # define all the relative maps that will be added to dictionary
    wall_map = np.zeros((31, 31))
    crate_map = np.zeros((31, 31))
    coin_map = np.zeros((31, 31))
    bomb_map_0 = np.zeros((31, 31))
    bomb_map_1 = np.zeros((31, 31))
    bomb_map_2 = np.zeros((31, 31))
    bomb_map_3 = np.zeros((31, 31))
    bomb_map_4 = np.zeros((31, 31))

    # calculate wall map, crate map and the bomb maps
    for x in range(len(field)):
        for y in range(len(field[x])):
            x_rel, y_rel = x - self_x, y - self_y
            if field[x, y] == -1:
                wall_map[15 + y_rel, 15 + x_rel] = 1
            if field[x, y] == 1:
                crate_map[15 + y_rel, 15 + x_rel] = 1
            for pos, time in bombs:
                if time == 0:
                    bomb_map_0[15 + y_rel, 15 + x_rel] = 1
                if time == 1:
                    bomb_map_1[15 + y_rel, 15 + x_rel] = 1
                if time == 2:
                    bomb_map_2[15 + y_rel, 15 + x_rel] = 1
                if time == 3:
                    bomb_map_3[15 + y_rel, 15 + x_rel] = 1
                if time == 4:
                    bomb_map_4[15 + y_rel, 15 + x_rel] = 1

    # calculate coin map
    for x, y in coins:
        x_rel, y_rel = x - self_x, y - self_y
        coin_map[15 + y_rel, 15 + x_rel] = 1

    relative_maps = {
        'wall_map': wall_map,
        'crate_map': crate_map,
        'coin_map': coin_map,
        'bomb_map_0': bomb_map_0,
        'bomb_map_1': bomb_map_1,
        'bomb_map_2': bomb_map_2,
        'bomb_map_3': bomb_map_3,
        'bomb_map_4': bomb_map_4
    }

    return relative_maps


def restrict_relative_map(relative_map, radius):
    """
    This function takes a relative map and reduces the size of it to the radius
    :param relative_map: a 2D numpy array, calculated before with get_relative_map()
    :param radius: an integer value with the look around og the agent
    :return: a 2D numpy array of the reduced array
    """

    index_min = 15 - radius
    index_max = 16 + radius
    relative_map = relative_map[index_min:index_max, index_min:index_max]

    return relative_map


"""
--------------------------------------------------------------------------------
shortest path features ---------------------------------------------------------
--------------------------------------------------------------------------------
"""


# For all shortest path features you need the Node class,
# because the A* search algorithm that is used, works on them:
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


# This algorithm is used to find the shortest path
# It only works if the class Node is defined
def shortest_path(free_tiles, start, target):
    """
    This is an A* search algorithm with heap queues to find the shortest path from start to target node
    :param free_tiles: free tiles is a 2D numpy array that contains TRUE if a tile is free or FALSE if not
    :param start: a (x,y) tuple with the position of the agent
    :param target: a (x,y) tuple with the position of the target
    :return: a list that contains the shortest path from start to target
    """

    start_node = Node(None, start)
    target_node = Node(None, target)
    start_node.g = start_node.h = start_node.f = 0
    target_node.g = target_node.h = target_node.f = 0

    open_nodes = []
    closed_nodes = []

    heapq.heappush(open_nodes, (start_node.f, start_node))
    free_tiles[target] = True

    while len(open_nodes) > 0:
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
            new_node = Node(current_node, position)
            neighbors.append(new_node)

        for neighbor in neighbors:
            if neighbor in closed_nodes:
                continue

            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - target_node.position[0]) ** 2) + (
                    (neighbor.position[1] - target_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            if not any(node[1] == neighbor for node in open_nodes):
                heapq.heappush(open_nodes, (neighbor.f, neighbor))
                continue

            for open_node in open_nodes:
                if neighbor == open_node[1]:
                    open_node[1].f = min(neighbor.f, open_node[1].f)
                    break

    return [start]


# This function returns the path to the nearest object in the objects list
# This requires the implementation of the class Node and the shortest_path algorithm
def get_nearest_object_path(free_tiles, pos, objects, offset=0):
    """
    This function finds the path that need the fewest steps from
    the agents current_node position to the nearest object from the object list
    :param free_tiles: a 2D numpy array of type bool, the entries that are True are positions the agent can walk along
    :param pos: a (x,y) tuple with the agents position
    :param objects: a list of positions of objects
    :param offset: an integer that gives a value that will be added to the size of the environment
    around the agent where he is looking at objects
    :return: a list refers to the the path from the agent to the nearest object
    """

    min_path = None
    min_path_val = np.infty

    if len(objects) == 0:
        return [pos]

    best_dist = min(np.sum(np.abs(np.subtract(objects, pos)), axis=1))
    near_objects = [elem for elem in objects if np.abs(elem[0] - pos[0]) + np.abs(elem[1] - pos[1]) <= best_dist +
                    offset]

    if len(near_objects) == 0:
        return [pos]

    for elem in near_objects:
        path = shortest_path(free_tiles, pos, elem)
        len_path = len(path)
        if len_path < min_path_val:
            min_path_val = len_path
            min_path = path

    return min_path


"""
--------------------------------------------------------------------------------
escape death features ----------------------------------------------------------
--------------------------------------------------------------------------------
"""


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
    return is_safe_death(pos, field, [(pos, 3)], look_ahead=False)
