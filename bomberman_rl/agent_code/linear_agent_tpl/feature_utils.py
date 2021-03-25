def get_nearest_coin_path(field, pos, coins, offset=0):
    """
    This function finds the path that need the fewest steps from
    the agents current_node position to the nearest coin
    :param offset: an integer that gives a value that will be added to the size of the environment
    around the agent where he is looking at coins
    :param field: a 2D numpy array of the field (empty, crates, walls)
    :param pos: a (x,y) tuple with the agents position
    :param coins: a 2D numpy array of the coin map
    :return: a list refers to the the path from the agent to the nearest coin
    """

    min_path = None
    min_path_val = 1000

    if len(coins) == 0:
        return [pos]

    best_dist = min(np.sum(np.abs(np.subtract(coins, pos)), axis=1))
    near_coins = [coin for coin in coins if np.abs(coin[0] - pos[0]) + np.abs(coin[1] - pos[1]) <= best_dist + offset]

    if len(near_coins) == 0:
        return [pos]

    for coin in near_coins:
        path = shortest_path(field, pos, coin)
        len_path = len(path)
        if len_path < min_path_val:
            min_path_val = len_path
            min_path = path

    return min_path


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
                heapq.heappush(open_nodes, (neighbor.f, neighbor))
                continue

            for open_node in open_nodes:
                if neighbor == open_node[1]:
                    open_node[1].f = min(neighbor.f, open_node[1].f)
                    break
