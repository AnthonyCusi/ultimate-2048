# This file implements the Advantage Actor-Critic (A2C) approach
# inspired by the implementation for CartPole at https://www.geeksforgeeks.org/actor-critic-algorithm-in-reinforcement-learning/

import numpy as np
import tensorflow as tf


class A2C:
    def __init__(self):

        self.state_shape = (4,4,1)
        self.gamma = 0.99
        self.moves = ['up', 'down', 'left', 'right']
        
        self.actor = self.build_actor()
        self.critic = self.build_critic()
        
        # will use these when I train the model
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
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(1))
        return model
    
    def train(self):
        '''Train the neural networks'''
    
    def next_action(self, game):
        '''Chooses next action by sampling from the actor network's output distribution'''

        if game.is_game_over():
            return None, 0
        
        state = game.get_state()
        state = np.expand_dims(state, axis=0)
        probs = self.actor(state).numpy()[0]
        p = probs.tolist()
        action_idx = p.index(max(p))
        move = self.moves[action_idx]
    
        return move, probs