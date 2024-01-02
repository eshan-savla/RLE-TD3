from collections import deque
import os
import numpy as np
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity=10000, agent_type=None):
        self.buffer = deque(maxlen=capacity)
        self.p_indices = [0.5 / 2] # Prioritized Experience Replay

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, np.expand_dims(reward, -1), next_state, np.expand_dims(done, -1)]) 

    def sample(self, batch_size=1, unbalance=0.8):
        p_indices = None
        if random.random() < unbalance:
            self.p_indices.extend((np.arange(len(self.buffer) - len(self.p_indices)) + 1)
                                  * 0.5 + self.p_indices[-1]) 
            p_indices = self.p_indices / np.sum(self.p_indices) # Priorisierung der Buffer Elemente neu berechnen
        sample_idx = np.random.choice(len(self.buffer),
                                      size=min(batch_size, len(self.buffer)),
                                      replace=False,
                                      p=p_indices) # Priorisierung neuer Elemente im Buffer
        sample = [self.buffer[s_i] for s_i in sample_idx]
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample)) # states, actions, rewards, next_states, dones aus dem Buffer holen
        return states, actions, rewards, next_states, dones

    def size(self):
        return len(self.buffer)
    
    def save(self, path:str):
        save_path = os.path.join(path, 'replay_buffer.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)
            pickle.dump(self.p_indices, f)
    
    def load(self, path:str):
        path = os.path.join(path, 'replay_buffer.pkl')
        with open(path, 'rb') as f:
            self.buffer = pickle.load(f)
            self.p_indices = pickle.load(f)