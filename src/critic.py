import tensorflow as tf

class Critic(tf.keras.layers.Layer):
    def __init__(self, state_units=(400, 300), action_units=(300,), units=(150,), stddev=0.00005, **kwargs):
        """
        Initializes the Critic class.

        Parameters:
            - state_units (tuple): A tuple specifying the number of units in each state layer.
            - action_units (tuple): A tuple specifying the number of units in each action layer.
            - units (tuple): A tuple specifying the number of units in each Q-value layer.
            - stddev (float): The standard deviation for the random normal initializer.
            - **kwargs: Additional keyword arguments to be passed to the base class constructor.

        Returns:
            - None
        """
        super(Critic, self).__init__(**kwargs)
    """_summary_:
    This class implements the Critic Network.
    The critic networks takes the states and actions as input and outputs the estimated q-values.
    """
    def __init__(self, state_units=(400, 300), action_units=(300,), units=(150,), stddev=0.00005, **kwargs): #initialize the Critic network with a default size of 400, 300, 150
        super(Critic, self).__init__(**kwargs) #initialize the super class
        self.layers_state = []
        for u in state_units: #for loop for the state units
            self.layers_state.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                           kernel_initializer=tf.keras.initializers.glorot_normal())) # create layers for the states

        self.layers_action = []
        for u in action_units: #for loop for the action units
            self.layers_action.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal()))  # create layers for the actions

        self.layers = []
        for u in units: #for loop for the units
            self.layers.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                     kernel_initializer=tf.keras.initializers.glorot_normal())) # create layers for the qvalues
        last_init = tf.random_normal_initializer(stddev=stddev)  # initialize the last layer with a random normal initializer
        self.layers.append(tf.keras.layers.Dense(1, kernel_initializer=last_init)) # letzte Layer welche den Q-Wert ausgibt 

        self.add = tf.keras.layers.Add()
         
    def call(self, inputs, **kwargs):
        """
        Forward pass of the critic network.

        Parameters:
            - inputs (dict): Dictionary containing the input tensors.
                - 'action': Tensor representing the action input.
                - 'state': Tensor representing the state input.

        Returns:
            - outputs (Tensor): Tensor representing the q-values output.
        """
        p_action = inputs['action']
        p_state = inputs['state']

        for l in self.layers_action: #for loop for the action layers
            p_action = l(p_action) # forward pass für die Actions

        for l in self.layers_state: #for loop for the state layers
            p_state = l(p_state) # forward pass für die States

        outputs = self.add([p_state, p_action])
        for l in self.layers:
            outputs = l(outputs) # forward pass for q-values

        return outputs # q-value
