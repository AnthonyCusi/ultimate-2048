# This file implements the Advantage Actor-Critic (A2C) approach
# inspired by the implementation for CartPole at https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/

import numpy as np
import tensorflow as tf
import copy


class A2C:
    def __init__(self):

        self.state_shape = (4,4,1)
        self.gamma = 0.99
        self.moves = ['up', 'down', 'left', 'right']
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def build_actor(self):
        '''Builds neural network for actor'''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(len(self.moves), activation='softmax'))
        return model
    
    def build_critic(self):
        '''Builds neural network for critic'''
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=self.state_shape))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model
    
    def next_action(self, game):
        '''Chooses next action by sampling from the actor network's output distribution'''

        if game.is_game_over():
            return None, 0
        
        state = game.get_state()
        state = np.expand_dims(state, axis=0)
        probs = self.actor(state).numpy()[0]
        p = probs.tolist()

        # sample for exploration
        action_idx = np.random.choice(len(self.moves), p=probs)
        move = self.moves[action_idx]
    
        return move, p
    
    def train(self, game, num_episodes=1000):
        '''Train the neural network'''

        for episode in range(num_episodes):
            # reset the current game state at the start of each episode
            training_game = copy.deepcopy(game)
            state = training_game.get_state()
            state = np.expand_dims(state, axis=-1)
            state = np.expand_dims(state, axis=0)
            total_reward = 0
            done = False

            # get next move, probabilities from actor
            move, probs = self.next_action(training_game)
            if move is None:
                break

            old_score = training_game.score
            # make move
            training_game.move(move)
            
            # reward based on improvement
            reward = training_game.score - old_score
            total_reward += reward
            
            # get new state
            next_state = training_game.get_state()