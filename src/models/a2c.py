# This file implements the Advantage Actor-Critic (A2C) approach
# inspired by the implementation for CartPole at https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/

import numpy as np
import tensorflow as tf
import copy
class A2C:
    def __init__(self, game):

        self.game = game
        self.state_shape = (4,4,1)
        self.gamma = 0.99
        self.moves = ['up', 'down', 'left', 'right']
        self.all_moves = ['up', 'down', 'left', 'right']
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

    def possible_moves(self):
        '''Returns possible moves from this game state'''
        moves = ['up', 'down', 'left', 'right']
        possible_moves = []
        for move in moves:
            test_game = copy.deepcopy(self.game)
            test_game.move(move)
            if not np.array_equal(self.game.board, test_game.board):
                possible_moves.append(move)
        return possible_moves
    
    def build_actor(self):
        '''Builds neural network for actor'''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Conv2D(32, (2,2), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(64, (2,2), activation='relu', padding='same'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(len(self.moves), activation='softmax'))
        return model
    
    def build_critic(self):
        '''Builds neural network for critic'''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Conv2D(32, (2,2), activation='relu', padding='same'))
        model.add(tf.keras.layers.Conv2D(64, (2,2), activation='relu', padding='same'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model
    
    def largest_tile_position_penalty(self, board):
        '''Penalizes largest tile being further from bottom left corner'''
        max_val = np.max(board)
        max_pos = np.where(board == max_val)
        
        # check if the largest tile is in the bottom left
        if len(max_pos[0]) > 0:
            row, col = max_pos[0][0], max_pos[1][0]
            
            # Manhattan distance from ideal position
            distance = abs(row - 3) + abs(col - 0)
            
            # higher penalty for greater distance
            return -distance * 10000
        return 0
    
    def second_largest_tile_position_penalty(self, board):
        '''Penalizes 2nd largest tile being further from spot above bottom left corner'''
        unique_values = np.unique(board)
        second_largest_value = sorted(unique_values, reverse=True)[1]
        second_largest_pos = np.argwhere(board == second_largest_value)[0]
        target_pos = (len(board) - 2, 0)
        # Manhattan distance    
        return -(abs(second_largest_pos[0] - target_pos[0]) + abs(second_largest_pos[1] - target_pos[1])) * 5000

    
    def next_action(self, game):
        '''Chooses next action by sampling from the actor network's output distribution'''

        if game.is_game_over():
            return None, 0
        
        self.moves = self.possible_moves()
        
        state = game.get_state()
        state = np.expand_dims(state, axis=0)
        probs = self.actor(state).numpy()[0]
        
        preferred_moves = []
        
        if 'down' in self.moves:
            preferred_moves.append('down')
        if 'left' in self.moves:
            preferred_moves.append('left')
        if preferred_moves:
            move = preferred_moves[np.random.choice(len(preferred_moves))]
            return move, probs
        
        # backup moves
        if 'up' in self.moves:
            move = 'up'
            return move, probs
        if 'right' in self.moves:
            move = 'right'
            return move, probs
        
        return self.moves[0], probs

        # # get probabilities of current possible moves so only they can be chosen from
        # probs_of_current_moves = []
        # indices_of_current_moves = []
        # for i in range(len(self.all_moves)):
        #     if self.all_moves[i]in self.moves:
        #         probs_of_current_moves.append(modified_probs[i])
        #         indices_of_current_moves.append(i)

        # if np.isnan(probs_of_current_moves).any() or sum(probs_of_current_moves) == 0:
        #     # if nan (numerical instability issues), then take max
        #     action_idx = probs_of_current_moves.index(max(probs_of_current_moves))
        # else:
        #     # otherwise, sample for exploration
        #     normalized_probs = np.array(probs_of_current_moves) / sum(probs_of_current_moves)
        #     action_idx = np.random.choice(len(self.moves), p=normalized_probs)
        
        # move = self.moves[action_idx]
        # return move, probs
    
    def update(self, state, action_idx, reward, next_state, game_board):
        '''Update actor and critic'''

        # add penalty for largest tile being far from corner
        position_reward = self.largest_tile_position_penalty(game_board)
        position_reward += self.second_largest_tile_position_penalty(game_board)
        reward += position_reward
        
        # compute the target for the critic
        next_value = self.critic(next_state)
        target = reward + self.gamma * next_value

        with tf.GradientTape(persistent=True) as tape_actor, tf.GradientTape(persistent=True) as tape_critic:
            
            # critic prediction
            value = self.critic(state)
            # calculate advantage based on prediction vs reality
            advantage = target - value

            probs = self.actor(state)

            # compute actor and critic losses
            entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10))
            actor_loss = -tf.math.log(probs[0, action_idx]) * advantage - 0.02 * entropy

            critic_loss = tf.square(advantage)

        # compute gradients and update networks
        actor_gradients = tape_actor.gradient(actor_loss, self.actor.trainable_variables)
        critic_gradients = tape_critic.gradient(critic_loss, self.critic.trainable_variables)

        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))


    def train(self, game, num_episodes=100):
        '''Train the neural networks'''

        for episode in range(num_episodes):
            # reset the current game state at the start of each episode
            training_game = copy.deepcopy(game)
            state = training_game.get_state()
            state = np.expand_dims(state, axis=-1)
            state = np.expand_dims(state, axis=0)
            total_reward = 0
            count = 0

            # num of moves to make
            while count < 10:
                # get next move, probabilities from actor
                move, probs = self.next_action(training_game)
                if move is None:
                    break

                old_score = training_game.score

                # make move
                training_game.move(move)

                # get new state
                next_state = training_game.get_state()
                
                # reward based on improvement
                reward = training_game.score - old_score
                total_reward += 2 * reward

                # preprocess next_state in the same way as state
                next_state_processed = np.expand_dims(next_state, axis=-1)
                next_state_processed = np.expand_dims(next_state_processed, axis=0)

                action_idx = self.moves.index(move)

                # update neural networks
                self.update(state, action_idx, reward, next_state_processed, training_game.board)

                # move to the next state
                state = next_state_processed

                count += 1