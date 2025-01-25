# This game rendering script is from https://github.com/scar17off/ai-2048.
# This file allows us to visualize the game and the steps that our models are taking.

import pygame

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

class GameRenderer:
    def __init__(self, width, height, grid_size, visual_mode=True):
        self.visual_mode = visual_mode
        if not self.visual_mode:
            return
            
        pygame.init()
        self.width = width
        self.height = height
        self.grid_size = grid_size
        self.screen = pygame.display.set_mode((self.width + 400, self.height))
        pygame.display.set_caption('2048 AI Training')
        self.cell_size = min(400 // max(self.grid_size), 50)
        self.board_offset = ((400 - (self.cell_size * self.grid_size[0])) // 2,
                           (400 - (self.cell_size * self.grid_size[1])) // 2)
        self.font = pygame.font.Font(None, min(36, self.cell_size))
        self.clock = pygame.time.Clock()

    def draw_game(self, game, moves_history):
        """Draw the current game state using pygame"""
        if not self.visual_mode:
            return
            
        self.screen.fill((250, 248, 239))
        
        # Draw game board
        for i in range(self.grid_size[0]):
            for j in range(self.grid_size[1]):
                value = game.board[i][j]
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
        
        # Draw stats below the board
        stats_y = self.board_offset[1] + self.grid_size[0] * self.cell_size + 20
        score_text = self.font.render(f'Score: {game.score}', True, (0, 0, 0))
        moves_text = self.font.render(f'Moves: {len(moves_history)}', True, (0, 0, 0))
        max_text = self.font.render(f'Max Tile: {game.board.max()}', True, (0, 0, 0))
        
        self.screen.blit(score_text, (10, stats_y))
        self.screen.blit(moves_text, (10, stats_y + 30))
        self.screen.blit(max_text, (200, stats_y))
        
        pygame.display.flip()
        self.clock.tick(20) # Limit to 20 FPS
        
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
        return True

    def draw_network_visualization(self, network_surface, position=(450, 50)):
        """Draw the network visualization on pygame surface"""
        if not self.visual_mode:
            return
        # Draw a white background for the visualization area
        pygame.draw.rect(self.screen, (255, 255, 255), 
                        (400, 0, 400, self.height))
        # Draw the network visualization
        self.screen.blit(network_surface, position)
        pygame.display.flip()

    def cleanup(self):
        """Clean up pygame resources"""
        if self.visual_mode:
            pygame.quit()
