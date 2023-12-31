from ddpg_config import cfg
from datetime import datetime
import os
import tensorflow as tf
import numpy as np
from critic import Critic
from actor import Actor
from noise import OUActionNoise

class DDPGAgent:
    def __init__(self, action_space, observation_shape, gamma=0.99, tau=0.001, epsilon=0.05):
        self.action_space = action_space
        self.tau = tau  # target network weight adaptation
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.save_dir = None
        
        self.actor = Actor(units=cfg.Actor.units, n_actions=action_space.shape[0], stddev=cfg.Actor.stddev) # Actor und Critic initialisieren
        self.critic = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.target_actor = Actor(units=cfg.Actor.units, n_actions=action_space.shape[0], stddev=cfg.Actor.stddev) # Target Actor und Critic initialisieren
        self.target_critic = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.DDPGAgent.learning_rate) # Optimizer für Actor und Critic
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.DDPGAgent.learning_rate)

        self.noise = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),
                                   std_deviation=float(cfg.OUNoiseOutput.sigma) * np.ones(1),theta=cfg.OUNoiseOutput.theta, dt=cfg.OUNoiseOutput.dt) # Noise für die Exploration

        self._init_networks(observation_shape)

    def setNoise(self, sigma, theta, dt):
        self.noise = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),std_deviation=float(sigma)*np.ones(1),theta=theta,dt=dt)

    def _init_networks(self, observation_shape):
        initial_state = np.zeros([1, observation_shape])

        initial_action = self.actor(initial_state) # Forward pass mit initialen Werten
        self.target_actor(initial_state)

        critic_input = {'action': initial_action, 'state': initial_state}
        self.critic(critic_input) # Forward pass mit initialen Werten
        self.target_critic(critic_input)

        self.target_actor.set_weights(self.actor.get_weights()) # Target Actor und Critic mit den initialen Werten initialisieren
        self.target_critic.set_weights(self.critic.get_weights())

    @staticmethod
    def update_target(model_target, model_ref, tau=0.0):
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                       list(zip(model_target.get_weights(), model_ref.get_weights()))] # zip macht aus zwei Listen eine Liste von Tupeln wo Elemente der gleichen Indizes zusammengefasst werden
        model_target.set_weights(new_weights)

    def act(self, observation, explore=True, random_action=False):
        if random_action or np.random.uniform(0, 1) < self.epsilon:
            a = self.action_space.sample()
        else:
            a = np.squeeze(self.actor(observation).numpy()) # sample action from policy
            if explore:
                a = np.add(a, self.noise()) # add noise for exploration
        a = np.clip(a, self.action_space.low, self.action_space.high) # setzt alle Wert größer als high auf high und alle kleiner als low auf low
        return a

    def compute_target_q(self, rewards, next_states, dones):
        actions = self.target_actor(next_states) # forward pass mit den nächsten States gibt die nächsten Actions gemäß der Policy aus target_actor
        critic_input = {'action': actions, 'state': next_states} 
        next_q = self.target_critic(critic_input) # forward pass mit den nächsten Actions und States gibt die nächsten Q-Werte aus target_critic
        target_q = rewards + (1 - dones) * next_q * self.gamma 
        return target_q 

    def get_actor_grads(self, states):
        with tf.GradientTape() as tape: # GradientTape speichert alle Operationen die auf Variablen ausgeführt werden
            actions = self.actor(states) # forward pass mit den States gibt die Actions gemäß der Policy aus actor nicht target_actor
            critic_input = {'action': actions, 'state': states}
            qs = self.critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            loss = -tf.math.reduce_mean(qs) # loss ist der negative Durchschnitt der Q-Werte. Ziel ist loss möglichst zu minimieren. Heißt negativere Zahl ist besser.
        gradients = tape.gradient(loss, self.actor.trainable_variables) # Gradienten berechnen, die loss nach den Gewichten und Biases ableiten. Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients] # Gradienten clippen um zu verhindern, dass die Gradienten zu groß werden
        return gradients, loss

    def get_critic_grads(self, states, actions, target_qs):
        with tf.GradientTape() as tape: 
            critic_input = {'action': actions, 'state': states} # kommt aus Replay Buffer
            qs = self.critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            loss = tf.reduce_mean(tf.abs(target_qs - qs)) # loss ist der Durchschnitt der absoluten Differenz zwischen den Q-Werten und den target Q-Werten. 
        gradients = tape.gradient(loss, self.critic.trainable_variables) # Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]
        return gradients, loss

    def learn(self, states, actions, rewards, next_states, dones):
        target_qs = self.compute_target_q(rewards, next_states, dones) # target Q-Werte berechnen aus Replay Buffer

        actor_grads, actor_loss = self.get_actor_grads(states)
        critic_grads, critic_loss = self.get_critic_grads(states, actions, target_qs)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables)) # Gewichte und Biases updaten anhand der Gradienten
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.target_update() # target Actor und Critic updaten
        return actor_loss, critic_loss

    def target_update(self):
        DDPGAgent.update_target(self.target_critic, self.critic, self.tau)
        DDPGAgent.update_target(self.target_actor, self.actor, self.tau)

    def save_weights(self):
        if self.save_dir is None:
            now = datetime.now()
            self.save_dir = now.strftime("%Y-%m-%d_%H-%M")
            os.makedirs(cfg.TD3Agent.weights_path + self.save_dir, exist_ok=True)
            self.save_dir = cfg.TD3Agent.weights_path + self.save_dir + "/"
        
        
        np.savez(self.save_dir + "ddpg_actor_weights", *self.actor.get_weights())
        np.savez(self.save_dir + "ddpg_critic_weights", *self.critic.get_weights())

        np.savez(self.save_dir + "ddpg_target_actor_weights", *self.target_actor.get_weights())
        np.savez(self.save_dir + "ddpg_target_critic_weights", *self.target_critic.get_weights())

    def load_weights(self, use_latest:bool=True, load_dir:str=None, lock_weights:bool=False):
        self.save_dir = load_dir
        if use_latest:
            load_dir = os.path.join(cfg.DDPGAgent.weights_path,max(os.listdir(cfg.TD3Agent.weights_path))) + "/"
        if lock_weights:
            if self.actor.trainable:
                print("Actor is trainable, setting to false. This is irreversible!")
                self.actor.trainable = False
            if self.critic.trainable:
                print("Critic is trainable, setting to false. This is irreversible!")
                self.critic.trainable = False
            if self.target_actor.trainable:
                print("Target Actor is trainable, setting to false. This is irreversible!")
                self.target_actor.trainable = False
            if self.target_critic.trainable:
                print("Target Critic is trainable, setting to false. This is irreversible!")
                self.target_critic.trainable = False

        actor_weights = (np.load(load_dir + "ddpg_actor_weights.npz"))
        critic_weights = np.load(load_dir + "ddpg_critic_weights.npz")
        target_actor_weights = np.load(load_dir + "ddpg_target_actor_weights.npz")
        target_critic_weights = np.load(load_dir + "ddpg_target_critic_weights.npz")

        self.actor.set_weights(list(actor_weights.values()))
        self.critic.set_weights(list(critic_weights.values()))
        self.target_actor.set_weights(list(target_actor_weights.values()))
        self.target_critic.set_weights(list(target_critic_weights.values()))
