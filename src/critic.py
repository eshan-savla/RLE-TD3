import tensorflow as tf

class Critic(tf.keras.layers.Layer):
    def __init__(self, state_units=(400, 300), action_units=(300,), units=(150,), stddev=0.00005, **kwargs):
        """
        Initializes the Critic class.

        Parameters:
            state_units (tuple): A tuple specifying the number of units in each state layer.
            action_units (tuple): A tuple specifying the number of units in each action layer.
            units (tuple): A tuple specifying the number of units in each Q-value layer.
            stddev (float): The standard deviation for the random normal initializer.
            **kwargs: Additional keyword arguments to be passed to the base class constructor.

        Returns:
            None
        """
        super(Critic, self).__init__(**kwargs)
        self.layers_state = []
        for u in state_units:
            self.layers_state.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal())) # Layers for states

        self.layers_action = []
        for u in action_units:
            self.layers_action.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal()))  # Layers for actions

        self.layers = []
        for u in units:
            self.layers.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                        kernel_initializer=tf.keras.initializers.glorot_normal())) # Layers for q-values
        last_init = tf.random_normal_initializer(stddev=stddev) 
        self.layers.append(tf.keras.layers.Dense(1, kernel_initializer=last_init)) # last layer which outputs the Q-value

        self.add = tf.keras.layers.Add()
         
    def call(self, inputs, **kwargs):
        """
        Forward pass of the critic network.

        Parameters:
            inputs (dict): Dictionary containing the input tensors.
                - 'action': Tensor representing the action input.
                - 'state': Tensor representing the state input.

        Returns:
            outputs (Tensor): Tensor representing the q-values output.
        """
        p_action = inputs['action']
        p_state = inputs['state']

        for l in self.layers_action:
            p_action = l(p_action) # forward pass for actions

        for l in self.layers_state:
            p_state = l(p_state) # forward pass for states

        outputs = self.add([p_state, p_action])
        for l in self.layers:
            outputs = l(outputs) # forward pass for q-values

        return outputs # q-value