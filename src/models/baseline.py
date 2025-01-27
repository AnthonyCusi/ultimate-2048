# This file implements the baseline model which selects random moves
import random
import numpy as np

def get_next_move(game_state, game_board):
    moves = ['up', 'down', 'left', 'right']
    # Choose a random order of moves 
    random.shuffle(moves)
    # Try moves in shuffled order
    for move in moves:
        # Create copy of game state
        test_board = game_board.copy()
        original = game_board.copy()
        # Try move; if successful then return it
        game_state.move(move)
        if not np.array_equal(original, game_state.board):
            return move, 1.0
    # No more moves are possible; game over
    return None, 0

