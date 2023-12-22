import tensorflow as tf

class Critic(tf.keras.layers.Layer):
    def __init__(self, state_units=(400, 300), action_units=(300,), units=(150,), **kwargs):
        super(Critic, self).__init__(**kwargs)
        self.layers_state = []
        for u in state_units:
            self.layers_state.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                           kernel_initializer=tf.keras.initializers.glorot_normal())) # Layers für die States

        self.layers_action = []
        for u in action_units:
            self.layers_action.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                            kernel_initializer=tf.keras.initializers.glorot_normal()))  # Layers für die Actions

        self.layers = []
        for u in units:
            self.layers.append(tf.keras.layers.Dense(u, activation=tf.nn.leaky_relu,
                                                     kernel_initializer=tf.keras.initializers.glorot_normal())) # Layers für die Q-Werte
        last_init = tf.random_normal_initializer(stddev=0.00005) 
        self.layers.append(tf.keras.layers.Dense(1, kernel_initializer=last_init)) # letzte Layer welche den Q-Wert ausgibt

        self.add = tf.keras.layers.Add()
         
    def call(self, inputs, **kwargs):
        p_action = inputs['action']
        p_state = inputs['state']

        for l in self.layers_action:
            p_action = l(p_action) # forward pass für die Actions

        for l in self.layers_state:
            p_state = l(p_state) # forward pass für die States

        outputs = self.add([p_state, p_action])
        for l in self.layers:
            outputs = l(outputs) # forward pass für die Q-Werte

        return outputs # Q-Wert