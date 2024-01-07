from td3_config import cfg
from hydra.utils import instantiate
import matplotlib.pyplot as plt
import pandas as pd
import os
import timeit
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from td3 import TD3Agent
from replay_buffer import ReplayBuffer
from functions import compute_avg_return

def main():
    load_replay_buffer = True
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    replay_buffer = instantiate(cfg.ReplayBuffer)
    env = gym.make('Ant-v3', max_episode_steps=1000,autoreset=True, render_mode='rgb_array') #human 
    agent = TD3Agent(env.action_space, env.observation_space.shape[0],gamma=cfg.TD3Agent.gamma,tau=cfg.TD3Agent.tau, epsilon=cfg.TD3Agent.epsilon, noise_clip=cfg.TD3Agent.noise_clip, policy_freq=cfg.TD3Agent.policy_freq)
    if type(cfg.TD3Agent.use_checkpoint_timestamp) == bool and cfg.TD3Agent.use_checkpoint_timestamp:
        print("Loading most recent checkpoint")
        agent.load_weights(use_latest=True)
        if load_replay_buffer:
            replay_buffer.load(agent.save_dir)
    elif not cfg.TD3Agent.use_checkpoint_timestamp:
        print("No checkpoint loaded. Starting from scratch.")
    else:
        print("Loading weights from timestamp: ", cfg.TD3Agent.use_checkpoint_timestamp)
        agent.load_weights(load_dir=os.path.join(cfg.TD3Agent.weights_path, cfg.TD3Agent.use_checkpoint_timestamp+'/'), use_latest=False)
        if load_replay_buffer:
            replay_buffer.load(agent.save_dir)
    total_timesteps = cfg.Training.start
    returns = list()
    actor_losses = list()
    critic1_losses = list() 
    critic2_losses = list()
    evals_dir = None
    first_training = True
    with tqdm(total=cfg.Training.timesteps, desc="Timesteps", position=total_timesteps, leave=True) as pbar:
        while total_timesteps <= cfg.Training.timesteps:
            obs, _ = env.reset()
            # gather experience
            agent.noise_output_net.reset()
            agent.noise_target_net.reset()

            ep_actor_loss = 0
            ep_critic1_loss = 0
            ep_critic2_loss = 0
            steps = 0
            for j in range(1000):
                steps += 1
                action = agent.act(np.array([obs]), random_action=(total_timesteps < cfg.Training.start_learning)) # i < 1 weil bei ersten Epoche keine Policy vorhanden ist
                # execute action

                # Patching terminated/truncated state behaviour based on issue:
                # https://github.com/Farama-Foundation/Gymnasium/pull/101
                # and
                # https://github.com/openai/gym/issues/3102
                new_obs, r, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                episode_truncated = not done or info.get("TimeLimit.truncated", False)
                info["TimeLimit.truncated"] = episode_truncated
                # truncated may have been set by the env too
                truncated = truncated or episode_truncated
                if done:
                    b = 0
                replay_buffer.put(obs, action, r, new_obs, done)
                obs = new_obs
                if done:
                    break
            total_timesteps += steps
            
            if total_timesteps >= cfg.Training.start_learning:      
            # Learn from the experiences in the replay buffer.
                for s in range(cfg.Training.batch_size):
                    s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(cfg.Training.sample_size, cfg.Training.unbalance)
                    actor_l, critic1_l, critic2_l = agent.learn(s_states, s_actions, s_rewards, s_next_states, s_dones,s)
                    ep_actor_loss += actor_l
                    ep_critic1_loss += critic1_l
                    ep_critic2_loss += critic2_l
                if total_timesteps % 25 == 0 or first_training:
                    first_training = False
                    avg_return, _ = compute_avg_return(env, agent, num_episodes=5, max_steps=1000, render=False)
                    print(
                        f'epoch {total_timesteps}, actor loss {ep_actor_loss / steps}, critic 1 loss {ep_critic1_loss / steps}, critic 2 loss {ep_critic2_loss/steps} , avg return {avg_return}')
                    agent.save_weights()
                    replay_buffer.save(agent.save_dir)
                if evals_dir is None:
                    evals_dir = '../evals/'+ agent.save_dir.split('/')[-2] + "/"
                    os.makedirs(evals_dir, exist_ok=True)   # create folder if not existing yet
                returns.append(avg_return)
                actor_losses.append(tf.get_static_value(ep_actor_loss) / steps)
                critic1_losses.append(tf.get_static_value(ep_critic1_loss) / steps)
                critic2_losses.append(tf.get_static_value(ep_critic2_loss) / steps)
                df = pd.DataFrame({'returns': returns, 'actor_losses': actor_losses, 'critic1_losses': critic1_losses, 'critic2_losses': critic2_losses})
                plot_losses = df.drop("returns", axis=1, inplace=False).plot(title='TD3 losses', figsize=(10, 5))
                plot_losses.set(xlabel='Epochs', ylabel='Loss')
                plot_losses.get_figure().savefig(evals_dir+'losses_td3.png')

                returns_df = pd.DataFrame({'returns': returns})
                plot_returns = returns_df.plot(title='TD3 returns', figsize=(10, 5))
                plot_returns.set(xlabel='Epochs', ylabel='Returns')
                plot_returns.get_figure().savefig(evals_dir+'returns_td3.png')
                plt.close('all')
                df.to_csv(evals_dir+'td3_results.csv', index=True) 
            
            pbar.update(steps)


    agent.save_weights()
    replay_buffer.save(agent.save_dir)
    
    env.close()

    
if __name__ == "__main__":
    elapsed_time = timeit.timeit(main, number=1)
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"The main function ran for {int(minutes)} minutes and {seconds:.2f} seconds.")
