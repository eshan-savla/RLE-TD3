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
        """
        Initializes the DDPG agent.

        Parameters:
            - action_space (gym.Space): The action space of the environment.
            - observation_shape (tuple): The shape of the observation space.
            - gamma (float, optional): The discount factor. Defaults to 0.99.
            - tau (float, optional): The target network weight adaptation factor. Defaults to 0.001.
            - epsilon (float, optional): The exploration factor. Defaults to 0.05.
        Returns:
            - None
        """
        self.action_space = action_space
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.save_dir = None
        
        self.actor = Actor(units=cfg.Actor.units, n_actions=action_space.shape[0], stddev=cfg.Actor.stddev)
        self.critic = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.target_actor = Actor(units=cfg.Actor.units, n_actions=action_space.shape[0], stddev=cfg.Actor.stddev)
        self.target_critic = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.DDPGAgent.learning_rate)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.DDPGAgent.learning_rate)

        self.noise = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),
                                std_deviation=float(cfg.OUNoiseOutput.sigma) * np.ones(1),theta=cfg.OUNoiseOutput.theta, dt=cfg.OUNoiseOutput.dt)

        self._init_networks(observation_shape)

    def setNoise(self, sigma, theta, dt):
        """
        Sets the noise parameters for the action noise.

        Parmeters:
            - sigma (float): The standard deviation of the noise.
            - theta (float): The rate of mean reversion for the noise.
            - dt (float): The time step size.

        Returns:
            - None
        """
        self.noise = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape), std_deviation=float(sigma)*np.ones(1), theta=theta, dt=dt)

    def _init_networks(self, observation_shape):
        """
        Initializes the actor and critic networks with the given observation shape.

        Parameters:
            - observation_shape (int): The shape of the observation input.

        Returns:
            - None
        """
        initial_state = np.zeros([1, observation_shape])

        initial_action = self.actor(initial_state) # Forward pass with initial values
        self.target_actor(initial_state)

        critic_input = {'action': initial_action, 'state': initial_state}
        self.critic(critic_input) # Forward pass wit initial values
        self.target_critic(critic_input)

        self.target_actor.set_weights(self.actor.get_weights()) # initialize target actor and critic with initial values
        self.target_critic.set_weights(self.critic.get_weights())

    @staticmethod
    def update_target(model_target, model_ref, tau=0.0):
        """
        Update the target model weights using a soft update strategy.

        Parameters:
            - model_target (tf.keras.Model): The target model to be updated.
            - model_ref (tf.keras.Model): The reference model whose weights will be used for the update.
            - tau (float): The interpolation parameter for the update. Default is 0.0, which means a hard update.

        Returns:
            - None
        """
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                    list(zip(model_target.get_weights(), model_ref.get_weights()))] # new weights are a linear combination of the target and reference weights
        model_target.set_weights(new_weights)

    def act(self, observation, explore=True, random_action=False):
        """
        Selects an action based on the given observation.

        Parameters:
            - observation (numpy.ndarray): The observation of the environment.
            - explore (bool): Whether to explore or exploit the policy. Default is True.
            - random_action (bool): Whether to select a random action. Default is False.

        Returns:
            - a (numpy.ndarray): The selected action.
        """
        if random_action or np.random.uniform(0, 1) < self.epsilon:
            a = self.action_space.sample()
        else:
            a = np.squeeze(self.actor(observation).numpy()) # sample action from policy
            if explore:
                a = np.add(a, self.noise()) # add noise for exploration
        a = np.clip(a, self.action_space.low, self.action_space.high) # clip action to action space
        return a

    def compute_target_q(self, rewards, next_states, dones):
        """
        Computes the target Q-values for the DDPG algorithm.

        Parameters:
            - rewards (torch.Tensor): The rewards received from the environment.
            - next_states (torch.Tensor): The next states observed from the environment.
            - dones (torch.Tensor): The done flags indicating whether the episode has terminated or truncated.

        Returns:
            - target_q (torch.Tensor): The target Q-values.
        """
        actions = self.target_actor(next_states)
        critic_input = {'action': actions, 'state': next_states}
        next_q = self.target_critic(critic_input) # forward pass with the next actions and states outputs the next Q values of target_critic
        target_q = rewards + (1 - dones) * next_q * self.gamma
        return target_q

    def get_actor_grads(self, states):
        """
        Calculates the gradients and the loss of the actor network with respect to the states.

        Parameters:
            - states (tf.Tensor): The input states.

        Returns:
            - gradients (List[tf.Tensor]): The gradients of the actor network.
            - loss (tf.Tensor): The loss value.
        """
        with tf.GradientTape() as tape: # GradientTape saves all operations performed on variables
            actions = self.actor(states) # forward pass with the states outputs the actions according to the policy actor not target_actor
            critic_input = {'action': actions, 'state': states}
            qs = self.critic(critic_input) # forward pass with the actions and states outputs the Q values critic not target_critic
            loss = -tf.math.reduce_mean(qs) # loss is the negative average of the Q values. The aim is to minimize loss as much as possible. Means a more negative number is better.
        gradients = tape.gradient(loss, self.actor.trainable_variables) # Calculate gradients, derive the loss according to the weights and biases. Partial derivation is used.
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients] # Clip gradients to prevent the gradients from becoming too large
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
        """
        Update the actor and critic networks based on the given batch of experiences.

        Parameters:
            - states (ndarray): The current states of the environment.
            - actions (ndarray): The actions taken in the current states.
            - rewards (ndarray): The rewards received for the actions taken.
            - next_states (ndarray): The next states of the environment.
            - dones (ndarray): The done flags indicating whether the episode is finished.

        Returns:
            - actor_loss (float): The loss value of the actor network.
            - critic_loss (float): The loss value of the critic network.
        """
        target_qs = self.compute_target_q(rewards, next_states, dones) # Calculate target Q values from replay buffer

        actor_grads, actor_loss = self.get_actor_grads(states)
        critic_grads, critic_loss = self.get_critic_grads(states, actions, target_qs)

        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables)) # Update weights and biases using the gradients
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        self.target_update() # update target actor und critic
        return actor_loss, critic_loss

    def target_update(self):
        """
        Update the target networks by copying the weights from the output networks.

        Parameters:
            - None

        Returns:
            - None
        """
        DDPGAgent.update_target(self.target_critic, self.critic, self.tau)
        DDPGAgent.update_target(self.target_actor, self.actor, self.tau)

    def save_weights(self):
        """
        Save the weights of the DDPG agent's networks.
        The weights are saved in separate files using the current timestamp as the directory name.

        Parameters:
            - None

        Returns:
            - None
        """
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
        """
        Loads the weights of the DDPG agent from a specified directory.

        Parameters:
            - use_latest (bool): If True, loads the weights from the latest directory in the weights_path.
            - load_dir (str): The directory path from which to load the weights.
            - lock_weights (bool): If True, locks the trainable status of the actor and critic models.

        Returns:
            - None
        """
        if use_latest:
            load_dir = os.path.join(cfg.DDPGAgent.weights_path,max(os.listdir(cfg.TD3Agent.weights_path))) + "/"
        self.save_dir = load_dir
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
