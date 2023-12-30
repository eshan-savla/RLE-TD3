from td3_config import cfg
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import pandas as pd
import os
import timeit
from tqdm import tqdm

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from td3 import TD3Agent
from replay_buffer import ReplayBuffer
from functions import compute_avg_return

def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    replay_buffer = instantiate(cfg.ReplayBuffer)
    env = gym.make('Ant-v3', ctrl_cost_weight=0.1, xml_file = "../models/ant.xml", render_mode='rgb_array') #human 
    agent = TD3Agent(env.action_space, env.observation_space.shape[0],gamma=cfg.TD3Agent.gamma,tau=cfg.TD3Agent.tau, epsilon=cfg.TD3Agent.epsilon, noise_clip=cfg.TD3Agent.noise_clip, policy_freq=cfg.TD3Agent.policy_freq)
    returns = list()
    actor_losses = list()
    critic1_losses = list() 
    critic2_losses = list()
    for i in tqdm(range(cfg.Training.epochs)):
        obs, _ = env.reset()
        # gather experience
        agent.noise_output_net.reset()
        agent.noise_target_net.reset()

        ep_actor_loss = 0
        ep_critic1_loss = 0
        ep_critic2_loss = 0
        steps = 0
        for j in range(cfg.Training.max_steps):
            steps += 1
            action = agent.act(np.array([obs]), random_action=(i < 1)) # i < 1 weil bei ersten Epoche keine Policy vorhanden ist
            # execute action
            new_obs, r, done, _, _ = env.step(action)
            replay_buffer.put(obs, action, r, new_obs, done)
            obs = new_obs
            if done:
                break
                
        # Learn from the experiences in the replay buffer.
        for s in range(cfg.Training.batch_size):
            s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(cfg.Training.sample_size, cfg.Training.unbalance)
            actor_l, critic1_l, critic2_l = agent.learn(s_states, s_actions, s_rewards, s_next_states, s_dones,s)
            ep_actor_loss += actor_l
            ep_critic1_loss += critic1_l
            ep_critic2_loss += critic2_l
        if i % 25 == 0:
            avg_return = compute_avg_return(env, agent, num_episodes=2, max_steps=cfg.Training.max_steps, render=False)
            print(
                f'epoch {i}, actor loss {ep_actor_loss / steps}, critic 1 loss {ep_critic1_loss / steps}, critic 2 loss {ep_critic2_loss/steps} , avg return {avg_return}')
            agent.save_weights()    
        returns.append(avg_return)
        actor_losses.append(tf.get_static_value(ep_actor_loss) / steps)
        critic1_losses.append(tf.get_static_value(ep_critic1_loss) / steps)
        critic2_losses.append(tf.get_static_value(ep_critic2_loss) / steps)

    agent.save_weights()
    df = pd.DataFrame({'returns': returns, 'actor_losses': actor_losses, 'critic1_losses': critic1_losses, 'critic2_losses': critic2_losses})
    
    os.makedirs('../evals/', exist_ok=True)     # create folder if not existing yet
    plot_losses = df.drop("returns", axis=1, inplace=False).plot(title='TD3 losses', figsize=(10, 5))
    plot_losses.set(xlabel='Training steps', ylabel='Loss')
    plot_losses.get_figure().savefig('../evals/losses_' + (agent.save_dir.split('/'))[-2] + '.png')

    returns_df = pd.DataFrame({'returns': returns})
    plot_returns = returns_df.plot(title='TD3 returns', figsize=(10, 5))
    plot_returns.set(xlabel='Training steps', ylabel='Returns')
    plot_returns.get_figure().savefig('../evals/returns_' + (agent.save_dir.split('/'))[-2] + '.png')
    
    df.to_csv('../evals/results_' + agent.save_dir.split('/')[-2] + '.csv', index=True)
    compute_avg_return(env, agent, num_episodes=10, max_steps=cfg.Training.max_steps, render=False)
    env.close()

    
if __name__ == "__main__":
    elapsed_time = timeit.timeit(main, number=1)
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"The main function ran for {int(minutes)} minutes and {seconds:.2f} seconds.")
