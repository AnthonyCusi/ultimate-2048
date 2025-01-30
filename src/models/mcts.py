# This file implements the Monte Carlo Tree Search (MCTS) approach, and 
# was inspired by the implementation for Tic-Tac-Toe at https://www.stephendiehl.com/posts/mtcs/
import numpy as np
import random
import copy

# Implements nodes that are part of the tree
class Node:
    def __init__(self, game_state, parent = None, last_move = None):
        self.game_state = game_state
        self.parent = parent
        self.last_move = last_move
        self.children = []
        self.visit_count = 0
        self.total_reward = 0

    def best_child(self, exploration_weight = 1):
        '''Selects the node with the largest number of visits'''
        try:
            #return max(self.children, key = lambda x: x.visit_count)
            return max(
                self.children,
                key = lambda child: child.total_reward / (child.visit_count + 1e-6) + \
                exploration_weight * np.sqrt(np.log(self.visit_count + 1) / (child.visit_count + 1e-6))
                - (100000 if child.last_move == "right" else 0)
                - (20000 if child.last_move == "up" else 0)
                - 100000 * self.distance_to_bottom_left(child.game_state.board)
                #+ 10000 * self.snake_score(child.game_state.board)
            )
        except:
            print(self.game_state.is_game_over())
            return(self)
        
    def distance_to_bottom_left(self, board):
        '''Calculates distance from largest tile to bottom left corner'''
        target_pos = (len(board) - 1, 0)
        max_pos = np.argwhere(board == np.max(board))[0]
        # Return the manhattan distance
        return abs(max_pos[0] - target_pos[0]) + abs(max_pos[1] - target_pos[1])
    
    def snake_pattern(self):
        '''Returns a list representing the ideal snake pattern'''
        indices = []
        for col in range(4):
            column_indices = [(row, col) for row in range(4)]
            if col % 2 == 0:
                indices.reverse()  # Reverse every other column for snake-like traversal
            indices.extend(column_indices)
        return indices
    
    def snake_score(self, board):
        '''Calculates the score for how well game is in snake pattern'''
        snake_indices = self.snake_pattern()
        flat_board = [(value, (i, j)) for i, row in enumerate(board) for j, value in enumerate(row)]
        flat_board.sort(reverse=True, key=lambda x: x[0])
        
        score = 0
        for (value, current), target in zip(flat_board, snake_indices):
            # Manhattan distance between current and target position
            dist = abs(current[0] - target[0]) + abs(current[1] - target[1])
            score -= dist * value * 10000
        return score

            
    def is_expanded(self):
        '''Returns bool based on if node is fully expanded'''
        return len(self.children) == len(self.possible_moves())

    def possible_moves(self):
        '''Returns possible moves from this game state'''
        moves = ['up', 'down', 'left', 'right']
        possible_moves = []
        for move in moves:
            test_state = copy.deepcopy(self.game_state)
            test_state.move(move)
            if not np.array_equal(self.game_state.board, test_state.board):
                possible_moves.append(move)
        return possible_moves
    
# Implements the Monte Carlo Tree Search Algorithm
class MCTS:
    def __init__(self, exploration_weight = 1, max_iterations = 20):
        self.exploration_weight = exploration_weight
        self.max_iterations = max_iterations

    def search(self, game_state):
        '''Performs the MCTS algorithm to return next best move'''
        root = Node(game_state = copy.deepcopy(game_state))
        for _ in range(self.max_iterations):
            node = self.select_next_move(root)
            reward = self.simulate_playout(node)
            self.backpropagate(node, reward)
        
        # Return move leading to the best next move
        best_child = root.best_child(exploration_weight = 0)
        return best_child.last_move, 1 # temporary 1 for confidence
    
    def select_next_move(self, node):
        '''Returns the next move from the current state'''
        while not node.game_state.is_game_over() and node.is_expanded():
            node = node.best_child(self.exploration_weight)
        if not node.game_state.is_game_over():
            return self.expand_node(node)
        return node
    
    def expand_node(self, node):
        '''Expands the current node'''
        attempted_moves = [child.last_move for child in node.children]
        for move in node.possible_moves():
            # Try not to move right if possible
            if move != "right" and move not in attempted_moves:
                # Attempt the next game state
                next_state = copy.deepcopy(node.game_state)
                next_state.move(move)
                child_node = Node(game_state = next_state, parent = node, last_move = move)
                node.children.append(child_node)
                return child_node
        # Move right if it's the last option
        for move in node.possible_moves():
            if move not in attempted_moves:
                next_state = copy.deepcopy(node.game_state)
                next_state.move(move)
                child_node = Node(game_state=next_state, parent=node, last_move=move)
                node.children.append(child_node)
                return child_node
        return node
    
    def simulate_playout(self, node):
        '''Simulates a random playout to estimate future outcomes'''
        current_state = copy.deepcopy(node.game_state)
        steps = 0
        reward = 0
        while not current_state.is_game_over() and steps < 100:
            move = random.choice(node.possible_moves())
            # Penalize going right
            if move == "right":
                reward -= 500
            else:
                current_state.move(move)
                reward += current_state.score
            steps += 1
        # Using score as reward
        return reward
    
    def backpropagate(self, node, reward):
        '''Updates rewards back up the tree'''
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent