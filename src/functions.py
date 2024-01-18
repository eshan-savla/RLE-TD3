import matplotlib.pyplot as plt
import numpy as np

def compute_avg_return(env, agent, num_episodes=1, max_steps=1000, render=False):
    """
    Computes the average return and standard deviation of returns for a given agent in a given environment.

    Parameters:
        env (object): The environment object.
        agent (object): The agent object.
        num_episodes (int, optional): The number of episodes to run. Defaults to 1.
        max_steps (int, optional): The maximum number of steps per episode. Defaults to 1000.
        render (bool, optional): Whether to render the environment. Defaults to False.

    Returns:
        avg_return (float): The average return.
        avg_return_stddev (float): The standard deviation of returns.
        episode_no (list): The episode numbers.
        returns (list): The returns for each episode.
        stddevs (list): The standard deviations of returns for each episode.
    """
    total_return = 0.0
    episode_no = []
    returns = []
    stddevs = []   
    
    max_steps = max_steps * num_episodes

    for e in range(num_episodes):   #iterate through multiple episodes

        obs, _ = env.reset()
        episode_return = 0.0
        terminated = False
        truncated = False
        steps = 0
        done = steps > max_steps
        while not (done):           #while steps < max_steps
            if render:              #render the environment if render = true (default = false)
                env.render()
            action = agent.act(np.array([obs]),explore=False)       #get action from agent network
            obs, r, terminated, truncated, info = env.step(action)  #execute action and get new observation, reward, terminated, truncated and info
            done = terminated or truncated                          #set done to True if action triggered termination or truncation
            if steps >= max_steps:            
                    episode_truncated = not done or info.get("TimeLimit.truncated", False)     
                    info["TimeLimit.truncated"] = episode_truncated
                    # truncated may have been set by the env too
                    truncated = truncated or episode_truncated
                    done = terminated or truncated
            episode_return += r     #add received reward to episode return
            steps += 1              #increment steps
        
        #append lists to calculate avg stddev and expand evaluation possibilities
        episode_no.append(e)                  #append episode number to episodes list
        returns.append(episode_return)      #add episode return to returns list each episode
        stddevs.append(np.std(returns))     #add standard deviation of returns to stddevs list each episode

        total_return += episode_return      #add episode return to total return each episode

        e += 1    #increment experiment number

    avg_return = total_return / num_episodes   #calculate average return of all experiments     
    avg_return_stddev = np.mean(stddevs)              #calculate average standard deviation of all experiments
        
    return avg_return, avg_return_stddev, episode_no, returns, stddevs


def flatten(lst):
    """
    Recursively flattens a nested list. -> Calculate the mean standard deviation per episode over all experiments

    Args:
        lst (list): The list to be flattened.

    Returns:
        result (list): The flattened list.
    """
    result = []
    for i in lst:
        if isinstance(i, list):
            result.extend(flatten(i))
        else:
            result.append(i)
    return result