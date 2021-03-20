import numpy as np

def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
   
    state_to_features(game_state) 
    
    return game_state['user_input']

def state_to_features(game_state: dict, readable = False) -> np.array:
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
  
    bombs = game_state['bombs']
    print(get_unsafe_positions(game_state['field'], bombs))
    
    return [0]

def get_unsafe_positions(field, bombs):
  unsafe_positions = []
  for bomb_pos, _ in bombs:
    unsafe_positions.append(bomb_pos)
    
    for x_offset in range(1, 4):
      pos = (bomb_pos[0] + x_offset, bomb_pos[1])
      if pos[0] > 16 or field[pos] == -1:
        break
      unsafe_positions.append(pos)
      
    for x_offset in range(-1, -4, -1):
      pos = (bomb_pos[0] + x_offset, bomb_pos[1])
      if pos[0] < 0 or field[pos] == -1:
        break
      unsafe_positions.append(pos)
      
    for y_offset in range(1, 4):
      pos = (bomb_pos[0], bomb_pos[1] + y_offset)
      if pos[0] > 16 or field[pos] == -1:
        break
      unsafe_positions.append(pos)
      
    for y_offset in range(-1, -4, -1):
      pos = (bomb_pos[0], bomb_pos[1] + y_offset)
      if pos[0] < 0 or field[pos] == -1:
        break
      unsafe_positions.append(pos)
   
  # TODO: Remove duplicates 
  return unsafe_positions
