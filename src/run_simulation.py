# This file is a modified version of the original file "ai_play.py" in https://github.com/scar17off/ai-2048/tree/main.
# The script in this file runs the 2048 simulation, and we made modifications in order
# to utilize our baseline and custom reinforcement learning models. 

import os
import tensorflow as tf
import random
import copy
import models.baseline as baseline
import models.mcts as mcts

# Try multiple GPU configuration approaches to avoid memory issues and improve performance
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Configure GPU memory growth
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Memory growth enabled for GPU: {gpu}")
        
        # Set TensorFlow to use the GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        print(f"Using GPU: {gpus[0]}")
        
        # Verify GPU is being used
        with tf.device('/GPU:0'):
            print("Testing GPU availability...")
            a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0]])
            print("GPU test result:", tf.matmul(a, b))
    else:
        print("No GPU devices found!")
except Exception as e:
    print(f"Error configuring GPU: {str(e)}")

COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
} 

# Rest of the imports
import numpy as np
import pygame
import sys
from game_logic import Game2048
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import json
import os

class AIGameGUI:
    def __init__(self):
        print("Initializing AI Game...")
        self.show_progress("Loading pygame", 0, 4)
        pygame.init()
        
        # Load configs
        self.show_progress("Loading configs", 1, 4)
        with open('config.json', 'r') as f:
            self.config = json.load(f)
        
        # Get grid size from config
        self.grid_size = self.config['game']['grid_size']
        self.width = 1000
        self.height = 600 + (100 if self.grid_size[0] > 4 else 0)
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('2048 AI Player')
        self.game = Game2048(config_dict=self.config)
        self.cell_size = min(700 // max(self.grid_size), 50)
        self.board_offset = ((400 - (self.cell_size * self.grid_size[0])) // 2,
                           (400 - (self.cell_size * self.grid_size[1])) // 2)
        self.font = pygame.font.Font(None, min(36, self.cell_size))
        
        # Load the AI model with progress bar
        #self.show_progress("Loading AI model", 2, 4)
        #print("\nLoading TensorFlow model...")
        #self.model = tf.keras.models.load_model('models/2048_model_final.h5', 
        #                                      custom_objects={'custom_loss': 'categorical_crossentropy'})
        #print("Model loaded successfully!")
        
        # Initialize visualization
        self.show_progress("Initializing visualization", 3, 4)
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.fig.patch.set_alpha(0.5)
        
        self.move_delay = 500
        self.last_move_time = pygame.time.get_ticks()
        self.moves_made = 0
        self.max_tile = 0
        
        self.show_progress("Initialization complete!", 4, 4)
        print("\nReady to play! Press SPACE to pause/resume, UP/DOWN to control speed.")

    def show_progress(self, message, current, total, width=50):
        """Show a progress bar in the console"""
        progress = float(current) / total
        filled = int(width * progress)
        bar = '=' * filled + '>' + '.' * (width - filled - 1)
        percentage = progress * 100
        
        sys.stdout.write(f'\r{message}: [{bar}] {percentage:.1f}%')
        sys.stdout.flush()
        
        if current == total:
            sys.stdout.write('\n')

    def get_ai_move(self, model_to_use):
        # # Prepare the board state for the model
        #state = self.game.board.copy()
        # state = np.where(state > 0, np.log2(state), 0).astype(np.float32)
        # state = state / 11.0
        # # Resize state to 4x4 for the model if necessary
        # if self.grid_size != [4, 4]:
        #     # Use max pooling to reduce larger boards to 4x4
        #     rows = np.array_split(state, 4, axis=0)
        #     reduced_state = np.zeros((4, 4))
        #     for i, row_group in enumerate(rows):
        #         cols = np.array_split(row_group, 4, axis=1)
        #         for j, block in enumerate(cols):
        #             reduced_state[i, j] = np.max(block)
        #     state = reduced_state
        
        # state = state.reshape(1, 4, 4, 1)
        
        # # Get model predictions
        # predictions = self.model.predict(state, verbose=0)[0]
        # self.update_network_visualization(predictions)
        
        # # Map predictions to moves
        # moves = ['up', 'down', 'left', 'right']
        # move_probs = list(zip(moves, predictions))
        # move_probs.sort(key=lambda x: x[1], reverse=True)
        
        # # Try moves in order of confidence until a valid one is found
        # for move, prob in move_probs:
        #     test_game = Game2048(config_dict=self.config)
        #     test_game.board = self.game.board.copy()
        #     original = test_game.board.copy()
        #     test_game.move(move)
        #     if not np.array_equal(original, test_game.board):
        #         return move, prob
        
        # return None, 0
        #moves = ['up', 'down', 'left', 'right']

        # Use the baseline model (selects random moves)
        if model_to_use == 'b':
            # Plotting random values for predicition confidence
            predictions = [random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)]
            self.update_network_visualization(predictions)
            # Get next move
            test_game = Game2048(config_dict = self.config)
            test_game.board = self.game.board.copy()
            return baseline.get_next_move(test_game, test_game.board)
        elif model_to_use == 'm':
            # --- TEMPORARY ---
            predictions = [random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1), random.uniform(0.5, 1)]
            self.update_network_visualization(predictions)
            # --- --- --- --- ---
            test_game = copy.deepcopy(self.game)
            mcts_model = mcts.MCTS()
            return mcts_model.search(test_game)
        else:
            pass # TO DO
        
 
        
    def update_network_visualization(self, predictions):
        self.ax.clear()
        moves = ['Up', 'Down', 'Left', 'Right']
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        
        # Create bar chart
        bars = self.ax.bar(moves, predictions, color=colors)
        
        # Customize the plot
        self.ax.set_ylim(0, 1)
        self.ax.set_title('AI Move Confidence')
        
        # Add percentage labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height*100:.1f}%',
                        ha='center', va='bottom')
        
        self.fig.canvas.draw()

    def draw_network_visualization(self):
        # Convert matplotlib figure to pygame surface
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_argb()
        size = canvas.get_width_height()
        
        # Create pygame surface
        surf = pygame.image.fromstring(raw_data, size, "ARGB")
        scaled_surf = pygame.transform.smoothscale(surf, (500, 500))

        # Draw on screen
        self.screen.blit(scaled_surf, (450, 50))

    def draw(self):
        self.screen.fill((250, 248, 239))
        
        # Draw game board
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                value = self.game.board[i][j]
                color = COLORS.get(value, (205, 193, 180))
                pygame.draw.rect(self.screen, color,
                               (self.board_offset[0] + j * self.cell_size, 
                                self.board_offset[1] + i * self.cell_size,
                                self.cell_size, self.cell_size))
                pygame.draw.rect(self.screen, (187, 173, 160),
                               (self.board_offset[0] + j * self.cell_size, 
                                self.board_offset[1] + i * self.cell_size,
                                self.cell_size, self.cell_size), 2)
                
                if value != 0:
                    text = self.font.render(str(value), True, (0, 0, 0))
                    text_rect = text.get_rect(center=(
                        self.board_offset[0] + j * self.cell_size + self.cell_size // 2,
                        self.board_offset[1] + i * self.cell_size + self.cell_size // 2
                    ))
                    self.screen.blit(text, text_rect)
        
        # Draw stats
        score_text = self.font.render(f'Score: {self.game.score}', True, (0, 0, 0))
        moves_text = self.font.render(f'Moves: {self.moves_made}', True, (0, 0, 0))
        max_text = self.font.render(f'Max Tile: {self.max_tile}', True, (0, 0, 0))
        
        stats_y = self.board_offset[1] + self.grid_size[0] * self.cell_size + 20
        self.screen.blit(score_text, (10, stats_y))
        self.screen.blit(moves_text, (10, stats_y + 30))
        self.screen.blit(max_text, (200, stats_y))
        
        # Draw neural network visualization
        self.draw_network_visualization()
        
        pygame.display.flip()

    def run(self, model_to_use):
        clock = pygame.time.Clock()
        running = True
        paused = False
        
        while running:
            current_time = pygame.time.get_ticks()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key == pygame.K_UP:
                        self.move_delay = max(100, self.move_delay - 100)
                    elif event.key == pygame.K_DOWN:
                        self.move_delay = min(2000, self.move_delay + 100)
            
            if not paused and current_time - self.last_move_time >= self.move_delay:
                move, confidence = self.get_ai_move(model_to_use)
                
                if move:
                    self.game.move(move)
                    self.moves_made += 1
                    self.max_tile = max(self.max_tile, np.max(self.game.board))
                    self.last_move_time = current_time
                else:
                    print("Game Over!")
                    running = False
            
            self.draw()
            clock.tick(60)
        
        pygame.quit()
        print(f"Final Score: {self.game.score}")
        print(f"Max Tile: {self.max_tile}")
        print(f"Moves Made: {self.moves_made}")

if __name__ == '__main__':
    game = AIGameGUI()
    print("----------------------------------------------")
    model_to_use = input("Enter model to run ('b' for baseline, 'm' for mcts): ")
    if model_to_use in ('b', 'm'):
        game.run(model_to_use)
    else:
        print('Invalid model selected.')