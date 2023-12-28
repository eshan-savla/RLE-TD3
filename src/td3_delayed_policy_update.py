import tensorflow as tf
from actor import Actor
from critic import Critic
import numpy as np
from noise import OUActionNoise


## Basis = added noise td3

class TD3Agent:
    def __init__(self, action_space, observation_shape, gamma=0.99, tau=0.0005, epsilon=0.05, policy_noise=0.2, noise_clip=0.5, policy_upd_freq=2):
        self.action_space = action_space
        self.tau = tau  # target network weight adaptation       ---> TODO: check if default value is correct with regard to the paper
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        
        #delayed policy update
        self.policy_upd_freq = policy_upd_freq   #default equals the recommendation of the paper: One policy update for every two q-function updates

        self.actor = Actor(n_actions=action_space.shape[0])
        self.critic_1 = Critic()
        self.critic_2 = Critic()

        # Initialize target networks as copies of the regular networks with their own weights and biases being updated slowly over time
        # The use of target networks introduces a delay in updating the actor network which helps to reduce the variance in the learning process
        # benefits: more stable and reliable q-value estimates during training process
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
        """_summary_:
        Double Q-Learning
        Applying the 2 critic network philosophy of TD3 to the target q-value calculation
        Using the minimum of both q-values (expected cumulative reward) to reduce the overestimation bias as a parameter of the neural network
           
        Args:
            rewards: reward for a given state-action pair
            next_states: next state after taking action in a given state
            dones: boolean value indicating if the episode is finished or not

        Returns:
            target_q: desired/ target estimate of q-value(cumulative reward) for a given state-action pair
        """
        next_action = np.clip(self.target_actor(next_states) + np.clip(self.noise(), -self.noise_clip, self.noise_clip), self.action_space.low, self.action_space.high)

        #Both critic networks need to be used to compute the q-value to reduce the overestimation bias by using the the minimum of both q-values
        critic_input_1 = {'action': next_action, 'state': next_states}
        critic_input_2 = {'action': next_action, 'state': next_states}
        next_q1 = self.target_critic_1(critic_input_1)
        next_q2 = self.target_critic_2(critic_input_2)

        #Evaluating both critics results and using the minimum of both
        next_q = np.minimum(next_q1, next_q2) #calculate the minimum to prevent overestimation bias
        target_q = rewards + (1 - dones) * next_q * self.gamma    #calulated by using the Bellman equation representing the expected cumulative reward that an agent can achieve by taking action in given state by following a policy
        
        return target_q

    def get_critic_grads(self, states, actions, target_qs, critic):

        """_summary_:
        Calcualation of gradients and loss for critic network by a compution q-values and estimated q-values.
        The loss is calucatted by the mean absolute difference between target Q-values and estimated q-values.
        The gradients are calculated by using the loss and the trainable variables of the critic network.
        They are clipped to prevent extreme values in rise/ falls of the gradients.

        Returns:
            gradients: gradients of the critic network
            loss: loss of the critic network
        """

        with tf.GradientTape() as tape: 
            critic_input = {'action': actions, 'state': states} # kommt aus Replay Buffer
            qs = critic(critic_input) # forward pass mit den Actions und States gibt die Q-Werte aus critic nicht target_critic
            loss = tf.reduce_mean(tf.abs(target_qs - qs)) # loss ist der Durchschnitt der absoluten Differenz zwischen den Q-Werten und den target Q-Werten. 
        gradients = tape.gradient(loss, critic.trainable_variables) # Partielle Ableitung
        gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]  #gradient clipping as part of Clipped Double Q-Learning
        return gradients, loss

    def get_actor_grads(self, states, target_critic):    #use target critic insted of regular critic to stabilize learning process and soften issue of overestimation bias
        """_summary_:
        Calculation of the gradients and losses for the actor network. 
        Computation of the action for given state using the actor network. 
        Computation of Q-values using critic network. 
        Calucation of loss as negative mean of Q-values.
        Compuation of the gradients of the loss with respect to the trainable variables of the actor network.

        Returns:
            gradients: gradients of the actor network (policy)
            loss: loss of the actor network (policy)
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
        if random_action or np.random.uniform(0, 1) < self.epsilon:
            a = self.action_space.sample()
        else:
            a = self.actor(observation).numpy()[:, 0] # sample action from policy
            if explore:
                a = np.squeeze([action + self.noise() for action in a]) # add noise for exploration
          
        a = np.clip(a, self.action_space.low, self.action_space.high) # setzt alle Wert größer als high auf high und alle kleiner als low auf low
        return a

    def learn(self, states, actions, rewards, next_states, dones, step):
        target_qs = self.compute_target_q(rewards, next_states, dones)
        critic_grads1, _ = self.get_critic_grads(states, actions, target_qs, self.critic_1)
        critic_grads2, _ = self.get_critic_grads(states, actions, target_qs, self.critic_2)
        self.critic_optimizer_1.apply_gradients(zip(critic_grads1, self.critic_1.trainable_variables))
        self.critic_optimizer_2.apply_gradients(zip(critic_grads2, self.critic_2.trainable_variables))

        if step % self.policy_upd_freq == 0:    # default: one update for every second q-function update
            actor_grads = self.get_actor_grads(states)
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))  #updates weights and biases of actor network by using the calculated gradients from the backpropagation
            self.target_update()
           
    def target_update(self):
        TD3Agent.update_target(self.target_actor, self.actor, self.tau)
        TD3Agent.update_target(self.target_critic_1, self.critic_1, self.tau)
        TD3Agent.update_target(self.target_critic_2, self.critic_2, self.tau)


    @staticmethod     
        # Decorator is used to define a method that belongs to the class itself rather than an instance of the class. 
        # It allows for the creation of utility functions or operations that are related to the class but do not require access to instance-specific data. 
        # Their methods can be called directly on the class without the need for an instance
    def update_target(model_target, model_ref, tau=0.005):
        """_summary_:
        Update of the target networks can be done periodically by slowly blending their weights with the weights of the regular networks.
        Slow updates smoothen the learning process and provides more consistent and accurate q-value estimates.
        
        Args:
            model_target: target network to be updated
            model_ref: regular network to be used as reference
            tau: blending factor, default value is 0.005      --TODO: check if default value is correct with regard to the paper
        """
        new_weights = [tau * ref_weight + (1 - tau) * target_weight for (target_weight, ref_weight) in
                       list(zip(model_target.get_weights(), model_ref.get_weights()))]
        model_target.set_weights(new_weights)


    