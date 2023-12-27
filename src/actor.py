import tensorflow as tf

class Actor(tf.keras.layers.Layer):
    def __init__(self, units=(400, 300), n_actions=2, stddev=0.00005, **kwargs):
        super(Actor, self).__init__(**kwargs)
        self.layers = []
        for i, u in enumerate(units):
            self.layers.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                     kernel_initializer=tf.keras.initializers.glorot_normal())) # Zwei layers mit neuronen
        last_init = tf.random_normal_initializer(stddev=stddev)
        self.layers.append(tf.keras.layers.Dense(n_actions, activation='tanh', kernel_initializer=last_init)) # letzte Layer welches die Aktionen ausgibt

    def call(self, inputs, **kwargs): # forward pass
        outputs = inputs
        for l in self.layers:
            outputs = l(outputs)
        return outputs