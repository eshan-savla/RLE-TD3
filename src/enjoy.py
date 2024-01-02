from ddpg import DDPGAgent
from td3 import TD3Agent
import os
import gymnasium as gym
from functions import compute_avg_return

def main(agent_type:str):
    load_dir = None
    use_latest = True
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    env = gym.make('Ant-v3', render_mode = "human") #, ctrl_cost_weight=0.1, xml_file = "../models/ant.xml", render_mode='human'
    if agent_type == "ddpg":
        from ddpg_config import cfg as ddpg_cfg
        agent = DDPGAgent(env.action_space, env.observation_space.shape[0],gamma=ddpg_cfg.DDPGAgent.gamma,tau=ddpg_cfg.DDPGAgent.tau, epsilon=0)
    elif agent_type == "td3":
        from td3_config import cfg as td3_cfg
        agent = TD3Agent(env.action_space, env.observation_space.shape[0],gamma=td3_cfg.TD3Agent.gamma,tau=td3_cfg.TD3Agent.tau, epsilon=0)
    else:
        raise ValueError("Invalid agent type")
    agent.load_weights(load_dir=load_dir, use_latest=use_latest)
    obs, _ = env.reset()
    print(f"Average return: {compute_avg_return(env, agent, num_episodes=20, max_steps=1000, render=False)}")
    env.close()

if __name__ == '__main__':
    main("td3")