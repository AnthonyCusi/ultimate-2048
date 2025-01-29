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
            )
        except:
            print(self.game_state.is_game_over())
            return(self)
    
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
    def __init__(self, exploration_weight = 1, max_iterations = 100):
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
            if move not in attempted_moves:
                # Attempt the next game state
                next_state = copy.deepcopy(node.game_state)
                next_state.move(move)
                child_node = Node(game_state = next_state, parent = node, last_move = move)
                node.children.append(child_node)
                return child_node
        return node
    
    def simulate_playout(self, node):
        '''Simulates a random playout to estimate future outcomes'''
        current_state = copy.deepcopy(node.game_state)
        steps = 0
        while not current_state.is_game_over() and steps < 100:
            move = random.choice(node.possible_moves())
            current_state.move(move)
            steps += 1
        # Using score as reward
        return current_state.score
    
    def backpropagate(self, node, reward):
        '''Updates rewards back up the tree'''
        while node is not None:
            node.visit_count += 1
            node.total_reward += reward
            node = node.parent