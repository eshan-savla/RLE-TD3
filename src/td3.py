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
        next_action = np.clip(self.target_actor(next_states) + np.clip(self.noise(), -self.noise_clip, self.noise_clip), self.action_space.low, self.action_space.high)

        critic_input_1 = {'action': next_action, 'state': next_states}
        critic_input_2 = {'action': next_action, 'state': next_states}
        next_q1 = self.target_critic_1(critic_input_1)
        next_q2 = self.target_critic_2(critic_input_2)
        next_q = np.minimum(next_q1, next_q2)
        target_q = rewards + (1 - dones) * next_q * self.gamma
        return target_q

    def get_critic_grads(self, states, actions, target_qs, critic):
        with tf.GradientTape() as tape: 
            critic_input = {'action': actions, 'state': states} # kommt aus Replay Buffer
            qs = critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            loss = tf.reduce_mean(tf.abs(target_qs - qs)) # loss ist der Durchschnitt der absoluten Differenz zwischen den Q-Werten und den target Q-Werten. 
        gradients = tape.gradient(loss, self.critic.trainable_variables) # Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        return gradients, loss

    def get_actor_grads(self, states):
        with tf.GradientTape() as tape: # GradientTape speichert alle Operationen die auf Variablen ausgeführt werden
            actions = self.actor(states) # forward pass mit den States gibt die Actions gemäß der Policy aus actor nicht target_actor
            critic_input = {'action': actions, 'state': states}
            qs = self.critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            loss = -tf.math.reduce_mean(qs) # loss ist der negative Durchschnitt der Q-Werte. Ziel ist loss möglichst zu minimieren. Heißt negativere Zahl ist besser.
        gradients = tape.gradient(loss, self.actor.trainable_variables) # Gradienten berechnen, die loss nach den Gewichten und Biases ableiten. Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients] # Gradienten clippen um zu verhindern, dass die Gradienten zu groß werden
        return gradients, loss
    
    def act(self, observation, explore=True, random_action=False):
        if random_action or np.random.uniform(0, 1) < self.epsilon:
            a = self.action_space.sample() # explore with random action
        else:
            a = self.actor(observation).numpy()[:, 0] # sample action from policy
            if explore:
                a += self.noise() # add noise for exploration
        a = np.clip(a, self.action_space.low, self.action_space.high) # setzt alle Wert größer als high auf high und alle kleiner als low auf low
        return a

    def learn(self, states, actions, rewards, next_states, dones, step):
        target_qs = self.compute_target_q(rewards, next_states, dones)
        critic_grads1, _ = self.get_critic_grads(states, actions, target_qs, self.critic_1)
        critic_grads2, _ = self.get_critic_grads(states, actions, target_qs, self.critic_2)
        self.critic_optimizer_1.apply_gradients(zip(critic_grads1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic_grads2, self.critic_2.trainable_variables))

        if step % self.policy_freq == 0:
            actor_grads = self.get_actor_grads(states)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.target_update()
           
    def target_update(self):
        TD3Agent.update_target(self.target_actor, self.actor, self.tau)
        TD3Agent.update_target(self.target_critic_1, self.critic_1, self.tau)
        TD3Agent.update_target(self.target_critic_2, self.critic_2, self.tau)


    @staticmethod
    def update_target(model_target, model_ref, tau=0.0):
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                       list(zip(model_target.get_weights(), model_ref.get_weights()))]
        model_target.set_weights(new_weights)