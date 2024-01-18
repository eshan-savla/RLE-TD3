import tensorflow as tf

class Actor(tf.keras.layers.Layer):
    def __init__(self, units=(400, 300), n_actions=2, stddev=0.00005, **kwargs):
            """
            Initialize the Actor network.

            Parameters:
                units (tuple): A tuple specifying the number of units/neurons in each hidden layer.
                n_actions (int): The dimension of the action space.
                stddev (float): The standard deviation for the kernel initializer.
                **kwargs: Additional keyword arguments to be passed to the base class.

            Returns:
                None
            """
            super(Actor, self).__init__(**kwargs)
            self.layers = []
            for i, u in enumerate(units):
                self.layers.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                         kernel_initializer=tf.keras.initializers.glorot_normal())) # two layers with neurons
            last_init = tf.random_normal_initializer(stddev=stddev)
            self.layers.append(tf.keras.layers.Dense(n_actions, activation='tanh', kernel_initializer=last_init)) # output layer with the dimension of the action space

    def call(self, inputs, **kwargs):
        """
        Perform a forward pass through the actor network.

        Parameters:
            inputs: The input tensor.
            **kwargs: Additional keyword arguments to be passed to the base class.

        Returns:
            outputs: The output tensor after passing through all the layers.
        """
        outputs = inputs
        for l in self.layers:
            outputs = l(outputs)
        return outputs