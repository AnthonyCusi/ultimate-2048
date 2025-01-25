# This Python implementation of the 2048 game is from https://github.com/scar17off/ai-2048.
# The file contains the logic of the base game which the models we train will interact with.

import numpy as np
import random
import json

class Game2048:
    def __init__(self, config_path=None, config_dict=None):
        # Load config either from file or dict
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            raise ValueError("Either config_path or config_dict must be provided")
        
        # Initialize board based on config
        self.grid_size = self.config['game']['grid_size']
        self.board = np.zeros(self.grid_size, dtype=int)
        self.score = 0
        self.win_tile = self.config['game']['win_tile']
        
        # Add initial tiles
        self._add_new_tile()
        self._add_new_tile()

    def _add_new_tile(self):
        empty_cells = [(i, j) for i in range(self.grid_size[0]) 
                      for j in range(self.grid_size[1]) if self.board[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            # Use spawn probabilities from config
            spawn_tiles = self.config['game']['spawn_tiles']
            value = np.random.choice(
                list(map(int, spawn_tiles.keys())),
                p=list(spawn_tiles.values())
            )
            self.board[i][j] = value

    def move(self, direction):
        original_board = self.board.copy()
        merged = set()

        if direction in ['left', 'right']:
            for i in range(self.grid_size[0]):
                self._merge_line(i, direction)
        else:
            self.board = self.board.T
            for i in range(self.grid_size[1]):
                self._merge_line(i, 'left' if direction == 'up' else 'right')
            self.board = self.board.T

        if not np.array_equal(original_board, self.board):
            self._add_new_tile()

    def _merge_line(self, i, direction):
        line = self.board[i].copy()
        if direction == 'right':
            line = line[::-1]

        # Remove zeros and merge
        non_zeros = [x for x in line if x != 0]
        merged = []
        j = 0
        while j < len(non_zeros):
            if j + 1 < len(non_zeros) and non_zeros[j] == non_zeros[j + 1]:
                merged.append(non_zeros[j] * 2)
                self.score += non_zeros[j] * 2
                j += 2
            else:
                merged.append(non_zeros[j])
                j += 1

        # Pad with zeros based on grid size
        merged.extend([0] * (self.grid_size[1] - len(merged)))
        
        if direction == 'right':
            merged = merged[::-1]
        
        self.board[i] = merged

    def is_game_over(self):
        if 0 in self.board:
            return False
        
        # Check horizontal merges
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1] - 1):
                if self.board[i][j] == self.board[i][j + 1]:
                    return False
        
        # Check vertical merges
        for i in range(self.grid_size[0] - 1):
            for j in range(self.grid_size[1]):
                if self.board[i][j] == self.board[i + 1][j]:
                    return False
        return True

    def get_state(self):
        return self.board.copy()
