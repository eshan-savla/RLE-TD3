import tensorflow as tf
from actor import Actor
from critic import Critic
import numpy as np
from noise import OUActionNoise


class TD3Agent:
    def __init__(self, action_space, observation_shape, gamma=0.99, tau=0.001, epsilon=0.05, policy_noise=0.2, noise_clip=0.5, policy_freq=2):
        self.action_space = action_space
        self.tau = tau  # target network weight adaptation
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.actor = Actor(n_actions=action_space.shape[0])
        self.critic_1 = Critic()
        self.critic_2 = Critic()

        self.target_actor = Actor(n_actions=action_space.shape[0])
        self.target_critic_1 = Critic()
        self.target_critic_2 = Critic()

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.001)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.noise = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),
                                   std_deviation=float(0.2) * np.ones(1))

        self._init_networks(observation_shape)

    def _init_networks(self, observation_shape):
        initial_state = np.zeros([1, observation_shape])

        initial_action = self.actor(initial_state)
        self.target_actor(initial_state)

        critic_input = {'action': initial_action, 'state': initial_state}
        self.critic_1(critic_input)
        self.critic_2(critic_input)
        self.target_critic_1(critic_input)
        self.target_critic_2(critic_input)

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic_1.set_weights(self.critic_1.get_weights())
        self.target_critic_2.set_weights(self.critic_2.get_weights())

    def compute_target_q(self, rewards, next_states, dones):
        noise = np.clip(np.random.normal(0, self.policy_noise, size=self.action_space.shape), -self.noise_clip, self.noise_clip)
        next_action = np.clip(self.target_actor(next_states) + noise, self.action_space.low, self.action_space.high)

        critic_input_1 = {'action': next_action, 'state': next_states}
        critic_input_2 = {'action': next_action, 'state': next_states}
        next_q1 = self.target_critic_1(critic_input_1)
        next_q2 = self.target_critic_2(critic_input_2)
        next_q = np.minimum(next_q1, next_q2)

        target_q = rewards + (1 - dones) * next_q * self.gamma
        return target_q

    def learn(self, states, actions, rewards, next_states, dones, step):
        target_qs = self.compute_target_q(rewards, next_states, dones)

        with tf.GradientTape() as tape1, tf.GradientTape() as tape2:
            critic_input_1 = {'action': actions, 'state': states}
            critic_input_2 = {'action': actions, 'state': states}
            qs1 = self.critic_1(critic_input_1)
            qs2 = self.critic_2(critic_input_2)
            critic_loss1 = tf.reduce_mean(tf.square(target_qs - qs1))
            critic_loss2 = tf.reduce_mean(tf.square(target_qs - qs2))

        critic_grads1 = tape1.gradient(critic_loss1, self.critic_1.trainable_variables)
        critic_grads2 = tape2.gradient(critic_loss2, self.critic_2.trainable_variables)

        self.critic_optimizer_1.apply_gradients(zip(critic_grads1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic_grads2, self.critic_2.trainable_variables))

        if step % self.policy_freq == 0:
            with tf.GradientTape() as tape:
                new_actions = self.actor(states)
                critic_input = {'action': new_actions, 'state': states}
                actor_loss = -self.critic_1(critic_input)
                actor_loss = tf.math.reduce_mean(actor_loss)

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

            self.update_target(self.target_actor, self.actor, self.tau)
            self.update_target(self.target_critic_1, self.critic_1, self.tau)
            self.update_target(self.target_critic_2, self.critic_2, self.tau)

    @staticmethod
    def update_target(model_target, model_ref, tau=0.0):
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                       list(zip(model_target.get_weights(), model_ref.get_weights()))]
        model_target.set_weights(new_weights)