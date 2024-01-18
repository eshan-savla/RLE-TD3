from collections import deque
import os
import numpy as np
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity=1000000, agent_type=None):     #initialize the replay buffer with a default capacity of 1.000.000
        self.buffer = deque(maxlen=capacity)    #create a double ended que with the initialized RB capacity
        self.p_indices = [0.5 / 2]              #prioritized experience replay

    def put(self, state, action, reward, next_state, done): 
        """_summary_:
        Add a new experience to the replay buffer
        """
        self.buffer.append([state, action, np.expand_dims(reward, -1), next_state, np.expand_dims(done, -1)])  #append the new experience (consisting of state, action, reward, next_state, done) to the buffer

    def sample(self, batch_size=1, unbalance=0.8):  
        """_summary_: Sample a batch of experiences from the replay buffer

        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            unbalance (float, optional): _description_. Defaults to 0.8.

        Returns:
            states, actions, rewards, next_states, dones from the replay buffer
        """

        p_indices = None
        
        if random.random() < unbalance: # Prioritizing the buffer elements
            self.p_indices.extend((np.arange(len(self.buffer) - len(self.p_indices)) + 1) 
                                  * 0.5 + self.p_indices[-1]) 
            
            p_indices = self.p_indices / np.sum(self.p_indices) # Calculate the prioritization of the buffer elements newly
        
        # Get the sample index from the buffer
        sample_idx = np.random.choice(len(self.buffer),
                                      size=min(batch_size, len(self.buffer)),
                                      replace=False,        # Sample without replacement
                                      p=p_indices)          # #prioritize new elements in the buffer
        
        sample = [self.buffer[s_i] for s_i in sample_idx]   # get the sample from the buffer
        states, actions, rewards, next_states, dones = map(np.array, zip(*sample)) # Get states, actions, rewards, next_states, dones for the chosen sample of the replay buffer
        
        return states, actions, rewards, next_states, dones

    def size(self):     #return the size of the replay buffer
        return len(self.buffer)
    
    def save(self, path:str): #save the replay buffer
        save_path = os.path.join(path, 'replay_buffer.pkl')
        with open(save_path, 'wb') as f: #save the replay buffer and the prioritization indices
            pickle.dump(self.buffer, f)
            pickle.dump(self.p_indices, f)
    
    def load(self, path:str): #load the replay buffer
        path = os.path.join(path, 'replay_buffer.pkl')
        with open(path, 'rb') as f: #load the replay buffer and the prioritization indices
            self.buffer = pickle.load(f)
            self.p_indices = pickle.load(f)