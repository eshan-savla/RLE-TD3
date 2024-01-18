import timeit
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



def main(load_replay_buffer:bool = True):
    """
    Main function for running the DDPG training algorithm.

    Parameters:
        - load_replay_buffer (bool): Flag indicating whether to load the replay buffer. Default is True.
    Returns:
        - None
    """
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    replay_buffer = instantiate(cfg.ReplayBuffer)
    env = gym.make('Ant-v3', max_episode_steps=1000,autoreset=True, render_mode='rgb_array') #human 
    agent = DDPGAgent(env.action_space, env.observation_space.shape[0],gamma=cfg.DDPGAgent.gamma,tau=cfg.DDPGAgent.tau, epsilon=cfg.DDPGAgent.epsilon)
    if type(cfg.DDPGAgent.use_checkpoint_timestamp) == bool and cfg.DDPGAgent.use_checkpoint_timestamp:
        print("Loading most recent checkpoint")
        agent.load_weights(use_latest=True)
        if load_replay_buffer:
            replay_buffer.load(agent.save_dir)
    elif not cfg.DDPGAgent.use_checkpoint_timestamp:
        print("No checkpoint loaded. Starting from scratch.")
    else:
        print("Loading weights from timestamp: ", cfg.DDPGAgent.use_checkpoint_timestamp)
        agent.load_weights(load_dir=os.path.join(cfg.DDPGAgent.weights_path, cfg.DDPGAgent.use_checkpoint_timestamp+'/'), use_latest=False)
        if load_replay_buffer:
            replay_buffer.load(agent.save_dir)
    total_timesteps = cfg.Training.start
    returns = list()
    actor_losses = list()
    critic_losses = list() 
    evals_dir = None
    first_training = True
    eval_count = 0
    with tqdm(total=cfg.Training.timesteps, desc="Timesteps", leave=True) as pbar:
        while total_timesteps <= cfg.Training.timesteps:
            obs, _ = env.reset()
            # gather experience
            agent.noise.reset()
            ep_actor_loss = 0
            ep_critic_loss = 0
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
                if steps >= 1000:
                    episode_truncated = not done or info.get("TimeLimit.truncated", False)
                    info["TimeLimit.truncated"] = episode_truncated
                    # truncated may have been set by the env too
                    truncated = truncated or episode_truncated
                    done = terminated or truncated
                replay_buffer.put(obs, action, r, new_obs, done)
                obs = new_obs
                if done:
                    break
               
            total_timesteps += steps
            
            if total_timesteps >= cfg.Training.start_learning:
                # Learn from the experiences in the replay buffer.
                for s in range(cfg.Training.batch_size):
                    s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(cfg.Training.sample_size, cfg.Training.unbalance)
                    actor_l, critic_l = agent.learn(s_states, s_actions, s_rewards, s_next_states, s_dones)
                    ep_actor_loss += actor_l
                    ep_critic_loss += critic_l
                if eval_count % 25 == 0 or first_training:
                    first_training = False
                    avg_return, _, _, _, _ = compute_avg_return(env, agent, num_episodes=5, max_steps=1000, render=False)
                    print(
                        f'Timestep {total_timesteps}, actor loss {ep_actor_loss / steps}, critic 1 loss {ep_critic_loss / steps} , avg return {avg_return}')
                    agent.save_weights()
                    replay_buffer.save(agent.save_dir)
                if evals_dir is None:
                    evals_dir = '../evals/'+ agent.save_dir.split('/')[-2] + "/"
                    os.makedirs(evals_dir, exist_ok=True)   # create folder if not existing yet
                returns.append(avg_return)
                actor_losses.append(tf.get_static_value(ep_actor_loss) / steps)
                critic_losses.append(tf.get_static_value(ep_critic_loss) / steps)
                df = pd.DataFrame({'returns': returns, 'actor_losses': actor_losses, 'critic_losses': critic_losses})
                plot_losses = df.drop("returns", axis=1, inplace=False).plot(title='DDPG losses', figsize=(10, 5), xticks=range(0, len(df), 25))
                plot_losses.set(xlabel='Timestamps', ylabel='Loss')
                plot_losses.get_figure().savefig(evals_dir+'losses_ddpg.png')

                returns_df = pd.DataFrame({'returns': returns})
                plot_returns = returns_df.plot(title='DDPG returns', figsize=(10, 5), xticks=range(0, len(df), 25))
                plot_returns.set(xlabel='Timestamps', ylabel='Returns')
                plot_returns.get_figure().savefig(evals_dir+'returns_ddpg.png')
                plt.close('all')
                df.to_csv(evals_dir+'ddpg_results.csv', index=True) 
                eval_count += 1
            pbar.update(steps)


    agent.save_weights()
    replay_buffer.save(agent.save_dir)
    
    env.close()

    
if __name__ == "__main__":
    elapsed_time = timeit.timeit(lambda: main(load_replay_buffer=True), number=1)
    minutes, seconds = divmod(elapsed_time, 60)
    print(f"The main function ran for {int(minutes)} minutes and {seconds:.2f} seconds.")