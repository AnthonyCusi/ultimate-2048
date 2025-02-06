
# This file implements the proximal policy optimization (PPO) approach
# inspired by the implementation of PPO at https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
import numpy as np
import tensorflow as tf
from keras import layers

class PPO:
    def __init__(self, state_shape=(4, 4, 1), action_space=4, gamma=0.99, lr=0.0003, clip_ratio=0.2):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        self.actor = self.build_actor()
        self.critic = self.build_critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    def build_actor(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_space, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def select_action(self, state):
        state = np.expand_dims(state, axis=(0, -1))
        probs = self.actor(state).numpy()[0]
        action = np.random.choice(self.action_space, p=probs)
        return action, probs[action]
    
    # @tf.function
    # def select_action(self, state):
    #     state = tf.expand_dims(state, axis=0)  # Add batch dimension
    #     state = tf.expand_dims(state, axis=-1) # Add channel dimension

    #     probs = self.actor(state)[0]
    #     action = tf.random.categorical(tf.math.log([probs]), 1)[0, 0]  # Sample action
    #     return action, probs[action]


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
            gae = delta + self.gamma * self.clip_ratio * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
        return np.array(advantages)

    def train(self, states, actions, old_probs, returns, advantages):
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)  # Normalize advantages

        with tf.GradientTape(persistent=True) as tape:
            probs = self.actor(states)
            action_probs = tf.reduce_sum(probs * tf.one_hot(actions, self.action_space), axis=1)

            ratios = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_probs + 1e-10))
            clipped_ratios = tf.clip_by_value(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio)

            policy_loss = -tf.reduce_mean(tf.minimum(ratios * advantages, clipped_ratios * advantages))
            entropy_bonus = tf.reduce_mean(-probs * tf.math.log(probs + 1e-10))  
            value_loss = tf.reduce_mean(tf.keras.losses.Huber()(returns, self.critic(states)))  # Huber loss for stability

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus 

        # Compute gradients based on total loss
        actor_grads = tape.gradient(total_loss, self.actor.trainable_variables)  
        critic_grads = tape.gradient(total_loss, self.critic.trainable_variables)  

        # Apply gradients
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        del tape  # Free memory


        
    def custom_reward(self, board):
        max_tile = np.max(board)
        empty_tiles = np.sum(board == 0)
        smoothness = -np.sum(np.abs(np.diff(np.sort(board.flatten()))))
        return max_tile + 0.5 * empty_tiles + 0.1 * smoothness
        
    def learn(self, game, episodes=1000, batch_size=64):
        for episode in range(episodes):
            state = game.get_state()
            done = False
            states, actions, rewards, dones, old_probs, values = [], [], [], [], [], []

            while not done:
                action, prob = self.select_action(state)
                #value = self.critic(np.expand_dims(state, axis=(0, -1))).numpy()[0, 0]
                value = self.critic(tf.expand_dims(state, axis=(0, -1)))[0, 0].numpy()


                game.move(['up', 'down', 'left', 'right'][action])
                next_state = game.get_state()
                # reward = np.max(game.board)
                reward = self.custom_reward(game.board)


                done = game.is_game_over()

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                old_probs.append(prob)
                values.append(value)

                state = next_state

            #next_value = self.critic(np.expand_dims(next_state, axis=(0, -1))).numpy()[0, 0]
            next_value = self.critic(tf.expand_dims(tf.expand_dims(next_state, axis=0), axis=-1))[0, 0].numpy()

            advantages = self.compute_advantages(rewards, values, next_value, dones)
            returns = advantages + np.array(values)

            self.train(np.array(states), np.array(actions), np.array(old_probs), np.array(returns), np.array(advantages))

