from td3_config import cfg
from datetime import datetime
import os

import tensorflow as tf
from actor import Actor
from critic import Critic
import numpy as np
from noise import OUActionNoise


## Basis = added noise td3

class TD3Agent:
    def __init__(self, action_space, observation_shape, gamma=0.99, tau=0.001, epsilon=0.05, noise_clip=0.5, policy_freq=2):
        """
        Initializes the TD3 agent.

        Parameters:
            - action_space (gym.Space): The action space of the environment.
            - observation_shape (tuple): The shape of the observation space.
            - gamma (float, optional): The discount factor. Defaults to 0.99.
            - tau (float, optional): The target network weight adaptation factor. Defaults to 0.001.
            - epsilon (float, optional): The exploration noise factor. Defaults to 0.05.
            - noise_clip (float, optional): The clipping range for the exploration noise. Defaults to 0.5.
            - policy_freq (int, optional): The frequency of updating the policy network. Defaults to 2.
        Returns:
            - None
        """
        self.action_space = action_space
        self.tau = tau
        self.gamma = gamma
        self.epsilon = epsilon
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.previous_actor_loss = 0
        self.save_dir = None

        self.actor = Actor(units=cfg.Actor.units,n_actions=action_space.shape[0], stddev=cfg.Actor.stddev)
        self.critic_1 = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)
        self.critic_2 = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.target_actor = Actor(units=cfg.Actor.units, n_actions=action_space.shape[0], stddev=cfg.Actor.stddev)
        self.target_critic_1 = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)
        self.target_critic_2 = Critic(state_units=cfg.Critic.state_units,action_units=cfg.Critic.action_units, units=cfg.Critic.units, stddev=cfg.Critic.stddev)

        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.TD3Agent.learning_rate)
        self.critic_optimizer_1 = tf.keras.optimizers.Adam(learning_rate=cfg.TD3Agent.learning_rate)
        self.critic_optimizer_2 = tf.keras.optimizers.Adam(learning_rate=cfg.TD3Agent.learning_rate)

        self.noise_output_net = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),
                                    std_deviation=float(cfg.OUNoiseOutput.sigma) * np.ones(1),theta=cfg.OUNoiseOutput.theta, dt=cfg.OUNoiseOutput.dt)
        
        self.noise_target_net = OUActionNoise(mean=np.zeros(np.array(self.action_space.sample()).shape),
                            std_deviation=float(cfg.OUNoiseTarget.sigma) * np.ones(1),theta=cfg.OUNoiseTarget.theta, dt=cfg.OUNoiseTarget.dt)

        self._init_networks(observation_shape)

    def _init_networks(self, observation_shape):
        """
        Initializes the actor and critic networks and sets their initial weights.

        Parameters:
            - observation_shape (int): The shape of the observation space.

        Returns:
            - None
        """
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


    def compute_target_q(self, rewards, next_states, dones, total_timesteps, reduce_noise: bool = False, noise_multiplicator:int = 5, min_noise_factor: float = 0.05): 
        """
        Computes the target Q-value for the TD3 algorithm.

        _summary_:Double Q-Learning:
        Applying the 2 critic network philosophy of TD3 to the target q-value calculation
        Using the minimum of both q-values (expected cumulative reward) to reduce the overestimation bias as a parameter of the neural network

        Parameters:
            - rewards (numpy.ndarray): Array of rewards for each transition.
            - next_states (numpy.ndarray): Array of next states for each transition.
            - dones (numpy.ndarray): Array of done flags for each transition.
            - total_timesteps (int): Total number of timesteps.
            - reduce_noise (bool, optional): Flag to reduce noise. Defaults to False.
            - noise_multiplicator (int, optional): Multiplicator for noise reduction. Defaults to 5.
            - min_noise_factor (float, optional): Minimum noise factor. Defaults to 0.05.

        Returns:
            - target_q (numpy.ndarray): Array of target Q-values for each transition.
        """
        factor = 1
        if reduce_noise:
            factor = np.add(-np.exp(total_timesteps/cfg.Training.timesteps * noise_multiplicator), np.exp(noise_multiplicator))
            factor = factor/np.exp(noise_multiplicator)
            if factor < min_noise_factor:
                factor = min_noise_factor
        next_action = np.clip(self.target_actor(next_states) + np.clip(factor*self.noise_target_net(), -self.noise_clip, self.noise_clip), self.action_space.low, self.action_space.high)

        #Both critic networks need to be used to compute the q-value to reduce the overestimation bias by using the the minimum of both q-values
        critic_input_1 = {'action': next_action, 'state': next_states}
        critic_input_2 = {'action': next_action, 'state': next_states}
        next_q1 = self.target_critic_1(critic_input_1)
        next_q2 = self.target_critic_2(critic_input_2)

        #Evaluating both critics results and using the minimum of both
        next_q = np.minimum(next_q1, next_q2) #calculate the minimum to prevent overestimation bias
        target_q = rewards + (1 - dones) * next_q * self.gamma    #calulated by using the Bellman equation representing the expected cumulative reward that an agent can achieve by taking action in given state by following a policy
        
        return target_q
    
    @staticmethod
    def get_actor_grads(self, states, target_critic):    #use target critic insted of regular critic to stabilize learning process and soften issue of overestimation bias
        """
        Calculates the gradients of the actor network with respect to the states and target critic.
        _summary_: Calcualation of gradients and loss for critic network by a compution q-values and estimated q-values.
        The loss is calucatted by the mean absolute difference between target Q-values and estimated q-values.
        The gradients are calculated by using the loss and the trainable variables of the critic network.
        They are clipped to prevent extreme values in rise/ falls of the gradients.

        Parameters:
            - states: The input states to the actor network.
            - target_critic: The target critic network used for stabilizing the learning process and reducing overestimation bias.

        Returns:
            - gradients: The gradients of the actor network with respect to the trainable variables.
            - loss: The negative mean of the Q-values calculated by the target critic network.
            """
        with tf.GradientTape() as tape: # GradientTape speichert alle Operationen die auf Variablen ausgeführt werden
            actions = self.actor(states) # forward pass mit den States gibt die Actions gemäß der Policy aus actor nicht target_actor
            critic_input = {'action': actions, 'state': states}
            
            qs = target_critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            
            loss = -tf.math.reduce_mean(qs) # loss ist der negative Durchschnitt der Q-Werte. Ziel ist loss möglichst zu minimieren. Heißt negativere Zahl ist besser.
        gradients = tape.gradient(loss, self.actor.trainable_variables) # Gradienten berechnen, die loss nach den Gewichten und Biases ableiten. Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients] # Gradienten clippen um zu verhindern, dass die Gradienten zu groß werden
        return gradients, loss
    
    def act(self, observation, explore=True, random_action=False):
        """
        Selects an action based on the given observation.

        Parameters:
            - observation (numpy.ndarray): The observation from the environment.
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
                a = np.add(a, self.noise_output_net()) # add noise for exploration
          
        a = np.clip(a, self.action_space.low, self.action_space.high) # setzt alle Wert größer als high auf high und alle kleiner als low auf low
        return a

    def learn(self, states, actions, rewards, next_states, dones, step, total_timesteps):
        """
        Update the actor and critic networks based on the given batch of experiences.

        Parameters:
            - states (array-like): The current states.
            - actions (array-like): The actions taken in the current states.
            - rewards (array-like): The rewards received for the actions taken.
            - next_states (array-like): The next states after taking the actions.
            - dones (array-like): Whether the episode is done after taking the actions.
            - step (int): The current step in the training process.
            - total_timesteps (int): The total number of timesteps in the training process.

        Returns:
            - previous_actor_loss (float): The loss of the previous actor update.
            - critics1_l (float): The loss of the first critic network.
            - critics2_l (float): The loss of the second critic network.
        """
        target_qs = self.compute_target_q(rewards, next_states, dones, total_timesteps)
        critic_grads1, critics1_l = self.get_critic_grads(states, actions, target_qs, self.critic_1)
        critic_grads2, critics2_l = self.get_critic_grads(states, actions, target_qs, self.critic_2)
        self.critic_optimizer_1.apply_gradients(zip(critic_grads1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic_grads2, self.critic_2.trainable_variables))

        if step % self.policy_freq == 0:
            actor_grads, self.previous_actor_loss = self.get_actor_grads(states, self.target_critic_1)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            self.target_update()
            
        return self.previous_actor_loss, critics1_l, critics2_l

    def target_update(self):
        """
        Update the target networks by copying the weights from the output networks.

        Parameters:
            - None

        Returns:
            - None
        """
        TD3Agent.update_target(self.target_actor, self.actor, self.tau)
        TD3Agent.update_target(self.target_critic_1, self.critic_1, self.tau)
        TD3Agent.update_target(self.target_critic_2, self.critic_2, self.tau)


    @staticmethod     
    # Decorator is used to define a method that belongs to the class itself rather than an instance of the class. 
    # It allows for the creation of utility functions or operations that are related to the class but do not require access to instance-specific data. 
    # Their methods can be called directly on the class without the need for an instance
    def update_target(model_target, model_ref, tau=0.005):
        """
        Update of the target networks can be done periodically by slowly blending their weights with the weights of the regular networks.
        Slow updates smoothen the learning process and provides more consistent and accurate q-value estimates.

        Parameters:
            - model_target (tf.keras.Model): The target model to be updated.
            - model_ref (tf.keras.Model): The reference model whose weights will be used for the update.
            - tau (float): The interpolation parameter for the update. Default is 0.0, which means a hard update.

        Returns:
            - None
        """
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                       list(zip(model_target.get_weights(), model_ref.get_weights()))]
        model_target.set_weights(new_weights)

    def save_weights(self):
        """
        Save the weights of the TD3 agent's networks.
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
        
        np.savez(self.save_dir + "actor_weights", *self.actor.get_weights())
        np.savez(self.save_dir + "critic1_weights", *self.critic_1.get_weights())
        np.savez(self.save_dir + "critic2_weights", *self.critic_2.get_weights())

        np.savez(self.save_dir + "target_actor_weights", *self.target_actor.get_weights())
        np.savez(self.save_dir + "target_critic1_weights", *self.target_critic_1.get_weights())
        np.savez(self.save_dir + "target_critic2_weights", *self.target_critic_2.get_weights())

    def load_weights(self, use_latest:bool=True, load_dir:str=None, lock_weights:bool=False):
        """
        Loads the weights of the TD3 agent from a specified directory.

        Parameters:
            - use_latest (bool): If True, loads the weights from the latest directory in the weights_path.
            - load_dir (str): The directory path from which to load the weights.
            - lock_weights (bool): If True, locks the trainable status of the actor and critic models.

        Returns:
            - None
        """
        if use_latest:
            load_dir = os.path.join(cfg.TD3Agent.weights_path,max(os.listdir(cfg.TD3Agent.weights_path))) + "/"
        self.save_dir = load_dir
        if lock_weights:
            if self.actor.trainable:
                print("Actor is trainable, setting to false. This is irreversible!")
                self.actor.trainable = False
            if self.critic_1.trainable:
                print("Critic 1 is trainable, setting to false. This is irreversible!")
                self.critic_1.trainable = False
            if self.critic_2.trainable:
                print("Critic 2 is trainable, setting to false. This is irreversible!")
                self.critic_2.trainable = False
            if self.target_actor.trainable:
                print("Target Actor is trainable, setting to false. This is irreversible!")
                self.target_actor.trainable = False
            if self.target_critic_1.trainable:
                print("Target Critic 1 is trainable, setting to false. This is irreversible!")
                self.target_critic_1.trainable = False
            if self.target_critic_2.trainable:
                print("Target Critic 2 is trainable, setting to false. This is irreversible!")
                self.target_critic_2.trainable = False

        self.actor.set_weights(list(np.load(load_dir + "actor_weights.npz").values()))
        self.critic_1.set_weights(list(np.load(load_dir + "critic1_weights.npz").values()))
        self.critic_2.set_weights(list(np.load(load_dir + "critic2_weights.npz").values()))

        self.target_actor.set_weights(list(np.load(load_dir + "target_actor_weights.npz").values()))
        self.target_critic_1.set_weights(list(np.load(load_dir + "target_critic1_weights.npz").values()))
        self.target_critic_2.set_weights(list(np.load(load_dir + "target_critic2_weights.npz").values()))
