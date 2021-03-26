"""
action_to_next_coin_features = np.array([0, 0, 0, 0])
next_pos = look_for_targets(free_tiles, pos, coins)
if next_pos is not None:
    # UP
    action_to_next_coin_features[0] = 1 if (pos[0], pos[1] - 1) == next_pos else 0
    # DOWN
    action_to_next_coin_features[1] = 1 if (pos[0], pos[1] + 1) == next_pos else 0
    # RIGHT
    action_to_next_coin_features[2] = 1 if (pos[0] + 1, pos[1]) == next_pos else 0
    # LEFT
    action_to_next_coin_features[3] = 1 if (pos[0] - 1, pos[1]) == next_pos else 0

"""


# the shortest path algorithm with the maze, way too slow BIG UFF
def shortest_path_maze(field, pos1, pos2):
    """
    This function finds the shortest path between two positions on the map
    :param field: a 2D numpy array of the field (empty, crates, walls)
    :param pos1: a (x,y) tuple with the agents position
    :param pos2: a (x,y) tuple with the destination position
    :return: a list that contains the shortest path from our agents to the destination position
    """

    if field[pos2] == 1:
        field[pos2] = 0
    # build up the maze
    maze = np.zeros(field.shape)
    maze[pos1] = 1

    def make_step(k):
        for i in range(len(maze)):
            for j in range(len(maze[i])):
                if maze[i][j] == k:
                    if i > 0 and maze[i - 1][j] == 0 and field[i - 1][j] == 0:
                        maze[i - 1][j] = k + 1
                    if j > 0 and maze[i][j - 1] == 0 and field[i][j - 1] == 0:
                        maze[i][j - 1] = k + 1
                    if i < len(maze) - 1 and maze[i + 1][j] == 0 and field[i + 1][j] == 0:
                        maze[i + 1][j] = k + 1
                    if j < len(maze[i]) - 1 and maze[i][j + 1] == 0 and field[i][j + 1] == 0:
                        maze[i][j + 1] = k + 1

    k = 0
    while maze[pos2[0]][pos2[1]] == 0:
        k += 1
        make_step(k)
        if k > 100:
            return None

    i, j = pos2
    k = maze[i][j]
    path = [(i, j)]
    while k > 1:
        if i > 0 and maze[i - 1][j] == k - 1:
            i, j = i - 1, j
            path.append((i, j))
            k -= 1
        elif j > 0 and maze[i][j - 1] == k - 1:
            i, j = i, j - 1
            path.append((i, j))
            k -= 1
        elif i < len(maze) - 1 and maze[i + 1][j] == k - 1:
            i, j = i + 1, j
            path.append((i, j))
            k -= 1
        elif j < len(maze[i]) - 1 and maze[i][j + 1] == k - 1:
            i, j = i, j + 1
            path.append((i, j))
            k -= 1

    return path[::-1]


# the improved shortest path algorithm without heaps
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

    open_nodes.append(start_node)

    while len(open_nodes) > 0:
        current_node = open_nodes[0]
        current_index = 0
        for index, item in enumerate(open_nodes):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_nodes.pop(current_index)
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

            # Create the f, g, and h values
            neighbor.g = current_node.g + 1
            neighbor.h = ((neighbor.position[0] - target_node.position[0]) ** 2) + (
                    (neighbor.position[1] - target_node.position[1]) ** 2)
            neighbor.f = neighbor.g + neighbor.h

            # neighbor is already in the open list
            if neighbor not in open_nodes:
                open_nodes.append(neighbor)
                continue

            for open_node in open_nodes:
                if neighbor == open_node:
                    open_node.g = min(neighbor.g, open_node.g)
                    break
