# Consider linear_agent_coin_callbacks
def get_relative_maps(game_state):
    # Return: dict mit keys
    # 'wall_map'
    # 'crate_map'
    # 'coin_map'
    # 'bomb_0_map'
    # ...
    # 'bomb_4_map'
    None

# helper functions
# get_relative_map_from_array(arr, pos)
# get_relative_map_from_position_lists(l, pos)

def restrict_relative_map(relative_map, radius):
    None

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