from collections import deque
import os
import numpy as np
import random
import pickle

class ReplayBuffer:
    def __init__(self, capacity=1000000, agent_type=None): #initialize the replay buffer with a default capacity of 1.000.000
        """
        Initialize the replay buffer.

        Parameters:
            - capacity (int): The maximum capacity of the replay buffer. Default is 1.000.000.
            - agent_type (str): The type of agent. Default is None.
        
        Returns:
            - None
        """
        self.buffer = deque(maxlen=capacity)    #create a double ended que with the initialized RB capacity
        self.p_indices = [0.5 / 2]              #prioritized experience replay

    def put(self, state, action, reward, next_state, done): 
        """
        Add a new experience to the buffer.

        Parameters:
            - state (np.ndarray): The current state.
            - action (int): The action taken.
            - reward (float): The reward received.
            - next_state (np.ndarray): The next state.
            - done (bool): Whether the episode is done or not.
        Returns:
            - None
        """
        self.buffer.append([state, action, np.expand_dims(reward, -1), next_state, np.expand_dims(done, -1)])  #append the new experience (consisting of state, action, reward, next_state, done) to the buffer

    def sample(self, batch_size=1, unbalance=0.8):
        """
        Randomly samples a batch of transitions from the replay buffer.

        Parameters:
            - batch_size (int): The number of transitions to sample. Default is 1.
            - unbalance (float): The probability of prioritizing newer buffer elements. Default is 0.8.

        Returns:
            - states (np.array): An array of states from the sampled transitions.
            - actions (np.array): An array of actions from the sampled transitions.
            - rewards (np.array): An array of rewards from the sampled transitions.
            - next_states (np.array): An array of next states from the sampled transitions.
            - dones (np.array): An array of done flags from the sampled transitions.
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

    def size(self):
        """
        Returns the size of the replay buffer.

        Parameters:
            - None

        Returns:
            - (int): The size of the replay buffer.
        """
        return len(self.buffer)
    
    def save(self, path:str):
        """
        Save the replay buffer and the prioritization indices to a file.

        Parameters:
            - path (str): The path where the file will be saved.

        Returns:
            - None
        """
        save_path = os.path.join(path, 'replay_buffer.pkl')
        with open(save_path, 'wb') as f: #save the replay buffer and the prioritization indices
            pickle.dump(self.buffer, f)
            pickle.dump(self.p_indices, f)
    
    def load(self, path:str):
        """
        Load the replay buffer from a file.

        Parameters:
            - path (str): The path to the file containing the replay buffer.

        Returns:
            - None
        """
        path = os.path.join(path, 'replay_buffer.pkl')
        with open(path, 'rb') as f: #load the replay buffer and the prioritization indices
            self.buffer = pickle.load(f)
            self.p_indices = pickle.load(f)