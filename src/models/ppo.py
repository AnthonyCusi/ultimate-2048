# This file implements the proximal policy optimization (PPO) approach
# inspired by the implementation of PPO at https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/

import numpy as np
import tensorflow as tf
from keras import layers
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PPO:
    def __init__(self, state_shape=(4, 4, 1), action_space=4, gamma=0.99, lr=0.0003, clip_ratio=0.2, gae_lambda=0.95):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.gae_lambda = gae_lambda
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # keep track of largest tiles to create snake pattern
        self.largest_tile_positions = []
        
        # define bottom left corner as target for largest tile
        self.target_position = (3, 0)
        
        # possible action indices
        self.UP = 0
        self.DOWN = 1
        self.LEFT = 2
        self.RIGHT = 3
        
        # movement penalties (discourages going right/up)
        self.right_penalty = 20.0  
        self.up_penalty = 5.0     

    def preprocess_state(self, state):
        # apply log normalization to reduce the large range of tile values
        state = np.log2(state + 1)
        # ensure the state has the correct shape (4,4,1)
        if state.ndim == 2:
            state = np.expand_dims(state, axis=-1)
        return state

    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            # use a convolutional layer to capture spatial patterns
            layers.Conv2D(128, kernel_size=2, activation='relu'),
            layers.Conv2D(64, kernel_size=2, activation='relu'),
            layers.Conv2D(64, kernel_size=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_space, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            # use a convolutional layer to capture spatial patterns
            layers.Conv2D(128, kernel_size=2, activation='relu'),
            layers.Conv2D(64, kernel_size=2, activation='relu'),
            layers.Conv2D(64, kernel_size=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(256, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])
        return model
    
    def get_largest_tile_position(self, board):
        max_val = np.max(board)
        positions = np.where(board == max_val)
        # if there are multiple largest tiles, take the first one
        return (positions[0][0], positions[1][0])
    
    def get_valid_moves(self, board):
        valid_moves = []
        # check each direction
        for action in range(self.action_space):
            temp_board = np.copy(board)
            move_name = ['up', 'down', 'left', 'right'][action]
            if self.simulate_move(temp_board, move_name):
                valid_moves.append(action)
                
        return valid_moves
    
    def evaluate_move_for_corner_strategy(self, board, action):
        temp_board = np.copy(board)
        move_name = ['up', 'down', 'left', 'right'][action]
        moved = self.simulate_move(temp_board, move_name)
        
        if not moved:
            return -100  # penalize invalid moves
        
        # get largest tile position after move
        largest_pos = self.get_largest_tile_position(temp_board)
        
        # Manhattan distance to target corner
        distance = abs(largest_pos[0] - self.target_position[0]) + abs(largest_pos[1] - self.target_position[1])
        corner_score = 10 - distance * 2
        
        # reward largest tile being in the corner
        if largest_pos == self.target_position:
            corner_score += 20
            
        if largest_pos == self.target_position:
            pattern_score = 0
            
            # check right neighbor
            if largest_pos[1] < 3 and temp_board[largest_pos[0], largest_pos[1]+1] <= temp_board[largest_pos]:
                pattern_score += 5
            
            # check upper neighbor
            if largest_pos[0] > 0 and temp_board[largest_pos[0]-1, largest_pos[1]] <= temp_board[largest_pos]:
                pattern_score += 5
                
            corner_score += pattern_score
        
        # penalty for moving right
        if action == self.RIGHT:
            valid_moves = self.get_valid_moves(board)
            if len(valid_moves) > 1: 
                corner_score -= self.right_penalty
        
        # penalty for moving up
        if action == self.UP:
            valid_moves = self.get_valid_moves(board)
            if len(valid_moves) > 1 and self.DOWN in valid_moves:
                corner_score -= self.up_penalty
        
        return corner_score
    
    def simulate_move(self, board, move):
        original_board = np.copy(board)
        
        if move == 'up':
            for j in range(4):
                # move all tiles up
                column = board[:, j]
                non_zeros = column[column != 0]
                column[:] = 0
                column[:len(non_zeros)] = non_zeros
                
                # merge tiles
                for i in range(3):
                    if column[i] != 0 and column[i] == column[i+1]:
                        column[i] *= 2
                        column[i+1:] = np.append(column[i+2:], 0)
                        
                board[:, j] = column
                
        elif move == 'down':
            for j in range(4):
                column = board[:, j]
                non_zeros = column[column != 0]
                column[:] = 0
                column[4-len(non_zeros):] = non_zeros
                
                for i in range(3, 0, -1):
                    if column[i] != 0 and column[i] == column[i-1]:
                        column[i] *= 2
                        column[:i] = np.append(0, column[:i-1])
                        
                board[:, j] = column
                
        elif move == 'left':
            for i in range(4):
                row = board[i, :]
                non_zeros = row[row != 0]
                row[:] = 0
                row[:len(non_zeros)] = non_zeros
                
                for j in range(3):
                    if row[j] != 0 and row[j] == row[j+1]:
                        row[j] *= 2
                        row[j+1:] = np.append(row[j+2:], 0)
                        
                board[i, :] = row
                
        elif move == 'right':
            for i in range(4):
                row = board[i, :]
                non_zeros = row[row != 0]
                row[:] = 0
                row[4-len(non_zeros):] = non_zeros
                
                # Merge tiles
                for j in range(3, 0, -1):
                    if row[j] != 0 and row[j] == row[j-1]:
                        row[j] *= 2
                        row[:j] = np.append(0, row[:j-1])
                        
                board[i, :] = row
        
        # check if the board changed
        return not np.array_equal(original_board, board)
        

    def select_action(self, state):
        # preprocess state to ensure consistent input shape and normalization
        processed_state = self.preprocess_state(state)
        state_batch = np.expand_dims(processed_state, axis=0)
        base_probs = self.actor(state_batch).numpy()[0]

        # valid moves
        valid_moves = self.get_valid_moves(state)
        if not valid_moves:
            action = np.random.randint(0, self.action_space)
            return action, base_probs[action]
        
        if valid_moves == [self.RIGHT]:
            return self.RIGHT, base_probs[self.RIGHT]
        if valid_moves == [self.UP]:
            return self.UP, base_probs[self.UP]
        
        # checking corner strategy
        corner_scores = np.zeros(self.action_space)
        for action in range(self.action_space):
            if action in valid_moves:
                corner_scores[action] = self.evaluate_move_for_corner_strategy(state, action)
            else:
                corner_scores[action] = -1000
        
        # additional penalties for up and right moves
        for action in valid_moves:
            if action == self.RIGHT:
                if len(valid_moves) > 1:
                    corner_scores[action] -= 50
            elif action == self.UP:
                if self.DOWN in valid_moves or self.LEFT in valid_moves:
                    corner_scores[action] -= 10
        
        # convert scores to probabilities using softmax
        exp_scores = np.exp(corner_scores - np.max(corner_scores))
        action_probs = exp_scores / np.sum(exp_scores)
        
        # ensure invalid moves have zero probability
        for i in range(self.action_space):
            if i not in valid_moves:
                action_probs[i] = 0
        
        # renormalize
        if np.sum(action_probs) > 0:
            action_probs = action_probs / np.sum(action_probs)
        else:
            action_probs = np.ones(self.action_space) / self.action_space
        
        # choose action based on the probabilities
        action = np.random.choice(self.action_space, p=action_probs)
        return action, base_probs[action]
    
    def next_action(self, game):
        state = game.get_state()
        action, confidence = self.select_action(state)
        move = ['up', 'down', 'left', 'right'][action]
        return move, confidence

    def compute_advantages(self, rewards, values, next_value, dones):
        advantages = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return np.array(advantages)

    def train(self, states, actions, old_probs, returns, advantages):
        # convert inputs to the correct types
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # normalize advantages
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_space), axis=1)

            ratios = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))
            entropy_bonus = tf.reduce_mean(-probs * tf.math.log(probs + 1e-10))
            value_loss = tf.reduce_mean(tf.keras.losses.Huber()(returns, self.critic(states)))

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus

        # compute gradients based on total loss
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)
        critic_grads = tape.gradient(total_loss, self.critic.trainable_variables)

        # apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        del tape

    def custom_reward(self, board, prev_board=None, action=None):
        # find the largest tile and its position
        max_tile = np.max(board)
        max_pos = self.get_largest_tile_position(board)
        
        empty_tiles = np.sum(board == 0)
        
        # Manhattan distance to bottom left
        distance_to_corner = abs(max_pos[0] - self.target_position[0]) + abs(max_pos[1] - self.target_position[1])
        corner_reward = 0
        if distance_to_corner == 0:
            corner_reward = max_tile * 5.0 
        else:
            corner_reward = -distance_to_corner * max_tile * 5 
        
        # encourage decreasing values from the corner
        pattern_reward = 0
        if max_pos == self.target_position:
            if max_pos[1] < 3 and board[max_pos[0], max_pos[1]+1] <= board[max_pos]:
                pattern_reward += max_tile * 0.2
            if max_pos[0] > 0 and board[max_pos[0]-1, max_pos[1]] <= board[max_pos]:
                pattern_reward += max_tile * 0.2
        
        merge_reward = 0
        if prev_board is not None:
            # get top 3 largest tiles from both boards
            current_top_tiles = np.sort(board.flatten())[-3:]  # top 3 largest tiles in current board
            prev_top_tiles = np.sort(prev_board.flatten())[-3:]  # top 3 largest tiles in previous board
            
            # check if largest tile increased
            if current_top_tiles[2] > prev_top_tiles[2]:
                merge_reward += (current_top_tiles[2] - prev_top_tiles[2]) * 25 
            
            # check if 2nd largest tile increased
            if current_top_tiles[1] > prev_top_tiles[1]:
                merge_reward += (current_top_tiles[1] - prev_top_tiles[1]) * 25 
            
            # check if 3rd largest tile increased
            if current_top_tiles[0] > prev_top_tiles[0]:
                merge_reward += (current_top_tiles[0] - prev_top_tiles[0]) * 3 
            
            for tile in current_top_tiles:
                if tile not in prev_top_tiles and tile > 4: 
                    merge_reward += tile * 0.5
        
        # movement penalties
        movement_penalty = 0
        if action is not None:
            # large penalty for moving right
            if action == self.RIGHT:
                valid_moves = self.get_valid_moves(prev_board)
                if len(valid_moves) > 1:  
                    movement_penalty -= max_tile * self.right_penalty
            
            # moderate penalty for moving up
            if action == self.UP:
                valid_moves = self.get_valid_moves(prev_board)
                if len(valid_moves) > 1 and self.DOWN in valid_moves:  
                    movement_penalty -= max_tile * self.up_penalty
        
        total_reward = (
            corner_reward * 10.0 +       
            pattern_reward * 5.0 +    
            empty_tiles * 1.0 +          
            merge_reward * 2.0 +         
            movement_penalty             
        )
        
        return total_reward
        
    def learn(self, game, episodes=1000):
        move_counts = {
            'up': 0,
            'down': 0,
            'left': 0,
            'right': 0
        }
        
        for episode in range(episodes):
            game.reset()  
            state = game.get_state()
            prev_board = np.copy(state)
            
            state = self.preprocess_state(state)
            done = False
            states, actions, rewards, dones, old_probs, values = [], [], [], [], [], []
            episode_reward = 0
            
            episode_moves = {
                'up': 0,
                'down': 0,
                'left': 0,
                'right': 0
            }
            
            step = 0
            while not done and step < 1000: 
                step += 1
                
                # Select action
                action, prob = self.select_action(prev_board)
                state_batch = np.expand_dims(state, axis=0)
                value = self.critic(state_batch)[0, 0].numpy()

                states.append(state)
                actions.append(action)
                old_probs.append(prob)
                values.append(value)

                move_name = ['up', 'down', 'left', 'right'][action]
                moved = game.move(move_name)
                
                if moved:
                    move_counts[move_name] += 1
                    episode_moves[move_name] += 1
                
                # get new state and board
                new_state = game.get_state()
                new_board = np.copy(new_state)
                
                reward = self.custom_reward(new_board, prev_board, action) if moved else -100
                rewards.append(reward)
                episode_reward += reward
                
                # track the position of the largest tile
                largest_pos = self.get_largest_tile_position(new_board)
                
                # check if game is over
                done = game.is_game_over()
                dones.append(done)

                # update state for next iteration
                state = self.preprocess_state(new_state)
                prev_board = new_board
            
            # store the final position of the largest tile
            self.largest_tile_positions.append(self.get_largest_tile_position(prev_board))
            
            if episode % 10 == 0:
                # Calculate percentage of episodes where largest tile ended in the corner
                if len(self.largest_tile_positions) > 0:
                    corner_success = sum(1 for pos in self.largest_tile_positions[-100:] 
                                        if pos == self.target_position) / min(100, len(self.largest_tile_positions))
                    corner_success_pct = corner_success * 100
                else:
                    corner_success_pct = 0
                
                total_episode_moves = sum(episode_moves.values())
                move_percentages = {
                    move: (count / total_episode_moves * 100) if total_episode_moves > 0 else 0
                    for move, count in episode_moves.items()
                }
                
                print(f"Episode {episode}, Reward: {episode_reward:.1f}, "
                      f"Max Tile: {np.max(prev_board)}, "
                      f"Corner Success: {corner_success_pct:.1f}%")
                print(f"  Moves: Up: {move_percentages['up']:.1f}%, "
                      f"Down: {move_percentages['down']:.1f}%, "
                      f"Left: {move_percentages['left']:.1f}%, "
                      f"Right: {move_percentages['right']:.1f}%")
            
            state_batch = np.expand_dims(state, axis=0)
            next_value = self.critic(state_batch)[0, 0].numpy()

            # compute advantages and returns
            advantages = self.compute_advantages(rewards, values, next_value, dones)
            returns = advantages + np.array(values)

            # train the agent
            self.train(np.array(states), np.array(actions), np.array(old_probs), np.array(returns), np.array(advantages))
            
            if episode % 50 == 0 and episode > 0:
                recent_corner_success = sum(1 for pos in self.largest_tile_positions[-50:] 
                                          if pos == self.target_position) / 50
                
                # calculate movement percentages
                total_moves = sum(move_counts.values())
                if total_moves > 0:
                    right_percentage = move_counts['right'] / total_moves * 100
                    up_percentage = move_counts['up'] / total_moves * 100
                    
                    # increase penalty if right moves happen too often
                    if right_percentage > 10:
                        self.right_penalty *= 1.5
                    
                    # increase penalty if up moves happen too often
                    if up_percentage > 20:
                        self.up_penalty *= 1.5
                
                # make the penalties stronger if highest tile isn't going in the corner
                if recent_corner_success < 0.5:
                    self.right_penalty *= 1.2
                    self.up_penalty *= 1.2
                
                move_counts = {move: 0 for move in move_counts}

    def save_weights(self, path):
        # ensure the target directory exists
        if not os.path.exists(path):
            os.makedirs(path)
        actor_path = os.path.join(path, "ppo_trained_actor.weights.h5")
        critic_path = os.path.join(path, "ppo_trained_critic.weights.h5")
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)

    def load_weights(self, path):
        # load weights for both actor and critic using the same file names as save_weights
        try:
            actor_path = os.path.join(path, "ppo_trained_actor.weights.h5")
            critic_path = os.path.join(path, "ppo_trained_critic.weights.h5")
            self.actor.load_weights(actor_path)
            self.critic.load_weights(critic_path)
            print("\nWeights loaded successfully")
        except Exception as e:
            print(f"Could not load weights: {e}")
            
if __name__ == '__main__':
    import json
    from game_logic import Game2048

    # load configuration from config.json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # create a game instance using the configuration
    game = Game2048(config_dict=config)
    
    # instantiate the PPO agent
    agent = PPO()
    
    print("Training PPO agent with strong corner strategy and movement restrictions...")
    # run the training loop
    agent.learn(game, episodes=1000)
    print("Training complete!")
    
    # save the trained weights to disk
    agent.save_weights("src/__pycache__")
    print("Weights saved")
    
    # test the trained agent
    print("\nTesting trained agent...")
    game.reset()
    steps = 0
    max_steps = 1000
    
    # track movements during testing
    test_moves = {'up': 0, 'down': 0, 'left': 0, 'right': 0}
    
    while not game.is_game_over() and steps < max_steps:
        move, _ = agent.next_action(game)
        moved = game.move(move)
        if moved:
            test_moves[move] += 1
        steps += 1
