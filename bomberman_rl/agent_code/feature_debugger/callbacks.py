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
   
    # print("TEST") 
    # print(get_reachable_states(game_state['self'][3], 0, game_state['field']))
    # print(get_reachable_states(game_state['self'][3], 1, game_state['field']))
    # print(get_reachable_states(game_state['self'][3], 2, game_state['field']))
    print(is_safe_death(game_state['self'][3], field, bombs))
    print(get_reachable_safe_tiles(game_state['self'][3], field, bombs))
    
    
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

def get_reachable_tiles(pos, num_steps, field):
  if num_steps == 0:
    return [pos]
  elif num_steps == 1:
    ret = [pos]
    pos_x, pos_y = pos
    
    pos_update = (pos_x + 1, pos_y) 
    ret.append(pos_update) if field[pos_update] == 0 else None
    
    pos_update = (pos_x - 1, pos_y) 
    ret.append(pos_update) if field[pos_update] == 0 else None
    
    pos_update = (pos_x, pos_y + 1) 
    ret.append(pos_update) if field[pos_update] == 0 else None
    
    pos_update = (pos_x, pos_y - 1) 
    ret.append(pos_update) if field[pos_update] == 0 else None
    
    return ret
  else:
    candidates = get_reachable_tiles(pos, num_steps - 1, field)
    ret = []
    for pos in candidates:
      ret.extend(x for x in get_reachable_tiles(pos, 1, field) if x not in ret)
    return ret
    
def get_reachable_safe_tiles(pos, field, bombs, timeoffset):
  if len(bombs) == 0:
    raise ValueError("No bombs placed.")
  
  reachable_tiles = set(get_reachable_tiles(pos, bombs[0][1] + 1 - timeoffset, field))
  unsafe_tiles = set(get_unsafe_positions(field, bombs))
  
  return [pos for pos in reachable_tiles if pos not in unsafe_tiles] 
  

def is_safe_death(pos, field, bombs):
  if len(bombs) == 0:
    return False
   
  reachable_tiles = set(get_reachable_tiles(pos, bombs[0][1] + 1, field))
  unsafe_tiles = set(get_unsafe_positions(field, bombs))
  
  return True if reachable_tiles.issubset(unsafe_tiles) else False
  
def is_safe_death_2(pos, field, bombs, timeoffset):
  if len(bombs) == 0:
    return False
  
  return len(get_reachable_safe_tiles(pos, field, bombs, timeoffset)) > 0)
