import pandas as pd
import matplotlib.pyplot as plt
import os
from ddpg_config import cfg
from hydra.utils import instantiate
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ddpg import DDPGAgent
from replay_buffer import ReplayBuffer
from functions import compute_avg_return
from tqdm import tqdm



def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    replay_buffer = instantiate(cfg.ReplayBuffer)
    env = gym.make('Ant-v3', render_mode="rgb_array")
    agent = DDPGAgent(env.action_space, env.observation_space.shape[0],gamma=cfg.DDPGAgent.gamma,tau=cfg.DDPGAgent.tau, epsilon=cfg.DDPGAgent.epsilon)
    # agent.setNoise(cfg.noise.sigma, cfg.noise.theta, cfg.noise.dt)
    if cfg.DDPGAgent.use_checkpoint_timestamp:
        agent.load_weights(use_latest=True)
        replay_buffer.load(agent.save_dir)
    elif not cfg.DDPGAgent.use_checkpoint_timestamp:
        pass
    else:
        agent.load_weights(load_dir=os.join(cfg.DDPGAgent.weights_path, cfg.DDPGAgent.use_checkpoint_timestamp), use_latest=False)
        replay_buffer.load(load_dir=os.join(cfg.DDPGAgent.weights_path, cfg.DDPGAgent.use_checkpoint_timestamp))
    returns = list()
    actor_losses = list()
    critic_losses = list()
    evals_dir = None
    for i in tqdm(range(cfg.Training.start, cfg.Training.epochs)):
        obs, _ = env.reset()
        # gather experience
        agent.noise.reset()
        ep_actor_loss = 0
        ep_critic_loss = 0
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
        for _ in range(cfg.Training.batch_size):
            s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(cfg.Training.sample_size, cfg.Training.unbalance)
            actor_l, critic_l = agent.learn(s_states, s_actions, s_rewards, s_next_states, s_dones)
            
            # sum up losses for the experience batch
            ep_actor_loss += actor_l # ep = episode
            ep_critic_loss += critic_l
            
        if i % 25 == 0 or i == cfg.Training.start:
            avg_return = compute_avg_return(env, agent, num_episodes=2, max_steps=cfg.Training.max_steps, render=False)
            print(
                f'epoch {i}, actor loss {ep_actor_loss / steps}, critic loss {ep_critic_loss / steps} , avg return {avg_return}')
            agent.save_weights()
            replay_buffer.save(agent.save_dir)
            if evals_dir is None:
                evals_dir = '../evals/'+ agent.save_dir.split('/')[-2] + "/"
                os.makedirs(evals_dir, exist_ok=True)   # create folder if not existing yet
            df = pd.DataFrame({'returns': returns, 'actor_losses': actor_losses, 'critic_losses': critic_losses})
            plot_losses = df.drop("returns", axis=1, inplace=False).plot(title='DDPG losses', figsize=(10, 5))
            plot_losses.set(xlabel='Epochs', ylabel='Loss')
            plot_losses.get_figure().savefig(evals_dir+'ddpg_losses.png')

            returns_df = pd.DataFrame({'returns': returns})
            plot_returns = returns_df.plot(title='DDPG returns', figsize=(10, 5))
            plot_returns.set(xlabel='Epochs', ylabel='Returns')
            plot_returns.get_figure().savefig(evals_dir+'ddpg_returns.png')
            plt.close('all')
            df.to_csv(evals_dir+'ddpg_results.csv', index=True)
        returns.append(avg_return)
        actor_losses.append(tf.get_static_value(ep_actor_loss) / steps)
        critic_losses.append(tf.get_static_value(ep_critic_loss) / steps)
    agent.save_weights()
    replay_buffer.save(agent.save_dir)   
    env.close()

if __name__ == "__main__":
    main()