
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
            layers.Conv2D(32, kernel_size=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(self.action_space, activation='softmax')
        ])
        return model

    def build_critic(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.state_shape),
            # use a convolutional layer to capture spatial patterns
            layers.Conv2D(32, kernel_size=2, activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(1)
        ])
        return model

    def select_action(self, state):
        # preprocess state to ensure consistent input shape and normalization
        processed_state = self.preprocess_state(state)
        state_batch = np.expand_dims(processed_state, axis=0)  # shape: (1, 4, 4, 1)
        probs = self.actor(state_batch).numpy()[0]
        action = np.random.choice(self.action_space, p=probs)
        return action, probs[action]
    
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

    # def train(self, states, actions, old_probs, returns, advantages):
        # normalize advantages for numerical stability
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
    def train(self, states, actions, old_probs, returns, advantages):
        # convert inputs to the correct dtypes
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        old_probs = tf.convert_to_tensor(old_probs, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        
        # normalize advantages for numerical stability
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


    def custom_reward(self, board):
        # calculate custom reward based on max tile, empty tiles, and smoothness
        max_tile = np.max(board)
        empty_tiles = np.sum(board == 0)
        smoothness = -np.sum(np.abs(np.diff(np.sort(board.flatten()))))
        return max_tile + 0.5 * empty_tiles + 0.1 * smoothness
        
    def learn(self, game, episodes=1000):
        for episode in range(episodes):
            state = game.get_state()
            # preprocess the state once at the start of the episode
            state = self.preprocess_state(state)
            done = False
            states, actions, rewards, dones, old_probs, values = [], [], [], [], [], []

            while not done:
                action, prob = self.select_action(state.squeeze())
                state_batch = np.expand_dims(state, axis=0)
                value = self.critic(state_batch)[0, 0].numpy()

                states.append(state)
                actions.append(action)
                old_probs.append(prob)
                values.append(value)

                game.move(['up', 'down', 'left', 'right'][action])
                reward = self.custom_reward(game.board)
                rewards.append(reward)
                done = game.is_game_over()
                dones.append(done)

                state = self.preprocess_state(game.get_state())

            # compute next_value from the terminal state
            state_batch = np.expand_dims(state, axis=0)
            next_value = self.critic(state_batch)[0, 0].numpy()

            advantages = self.compute_advantages(rewards, values, next_value, dones)
            returns = advantages + np.array(values)

            self.train(np.array(states), np.array(actions), np.array(old_probs), np.array(returns), np.array(advantages))

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
            print("\nweights loaded successfully")
        except Exception as e:
            print(f"could not load weights: {e}")
            
if __name__ == '__main__':
    import json
    from game_logic import Game2048

    # load configuration from config.json
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # create a game instance using the configuration
    game = Game2048(config_dict=config)
    
    # instantiate the ppo agent
    agent = PPO()
    
    print("training ppo agent...")
    # run the training loop for 1000 episodes
    agent.learn(game, episodes=1000)
    print("training complete!")
    
    # save the trained weights to disk in the desired directory
    agent.save_weights("src/__pycache__")
    print("weights saved")

