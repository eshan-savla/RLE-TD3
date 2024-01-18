from ddpg import DDPGAgent
from td3 import TD3Agent
import os
import gymnasium as gym
from functions import compute_avg_return
import pandas as pd

# Specify the path to the to be evaluated model

load_dir    = None     #Specify the path to the model you want to evaluate. For more information: see Mapping_mod-conf.md
use_latest  = True   #True, if you want to use the latest checkpoint of trained models
    

def enjoy(agent_type:str, load_dir:str=None, use_latest:str=True, render_mode:str=None):  #defaults: agent_type="td3", load_dir=None, use_latest=True, render_mode=None

    os.chdir(os.path.dirname(os.path.abspath(__file__)))                    # change directory to the directory of this file
    env = gym.make(id='Ant-v3', autoreset=True, render_mode = render_mode)  # create the environment 
                                                                                # id = Environment ID 
                                                                                # autoreset=True => automatically reset the environment after an episode is done
                                                                                #render_mode='human' => render the environment visually // render_mode='rgb_array' => render the environment as an array to collect results
    if agent_type == "ddpg":   #if you want to enjoy a DDPG agent
        from ddpg_config import cfg as ddpg_cfg, config_name
        #create a DDPGAgent with the same parameters as the one used for training and configurations specified in yaml file (loaded via Hydra)
        agent = DDPGAgent(env.action_space, env.observation_space.shape[0],gamma=ddpg_cfg.DDPGAgent.gamma,tau=ddpg_cfg.DDPGAgent.tau, epsilon=0) 
    
    elif agent_type == "td3":  #if you want to enjoy a TD3 agent
        #create a TD3Agent with the same parameters as the one used for training and configurations specified in yaml file (loaded via Hydra)
        from td3_config import cfg as td3_cfg, config_name
        agent = TD3Agent(env.action_space, env.observation_space.shape[0],gamma=td3_cfg.TD3Agent.gamma,tau=td3_cfg.TD3Agent.tau, epsilon=0)
    else: #handling wrong agent type specification
        raise ValueError("Invalid agent type")
    
    agent.load_weights(load_dir=load_dir, use_latest=use_latest) #load the weights of the agent
    obs, _ = env.reset() #reset the environment and get the initial observation
    
    avg_return, avg_return_stddev, episode_no, returns, stddevs = compute_avg_return(env, agent, num_episodes=150, max_steps=1000, render=False) #compte the average return and specify the to be evaluated number of episodes
    print(f"Average return: {avg_return}, Standard deviation: {avg_return_stddev}")
    
    #To get a unique benchmark result, we save the results in a csv file
    time_stamp = agent.save_dir.split("/")[-2]  #get timestamp 
    user_name = os.getlogin()                   #get username 
    df = pd.DataFrame({"file": [config_name], "time_stamp": [time_stamp], "user_name": [user_name], "agent_type": [agent_type], "avg_return": [avg_return], "return_stddev": [avg_return_stddev], "episode_no": [episode_no], "returns": [returns], "stddevs": [stddevs]}) #create pandas DF
    df.to_csv("../benchmarks_new.csv", mode="a", header=False, index=False)
    env.close()

if __name__ == '__main__':
    enjoy("td3")             #speficy the desired algorithm/ agent type ("ddpg" or "td3")