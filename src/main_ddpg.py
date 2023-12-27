from src.ddpg_config import cfg
from hydra.utils import instantiate
import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from ddpg import DDPGAgent
from replay_buffer import ReplayBuffer
from functions import compute_avg_return



def main():
    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    replay_buffer = instantiate(cfg.ReplayBuffer)
    env = gym.make('Pendulum-v1', render_mode='rgb_array')
    agent = DDPGAgent(env.action_space, env.observation_space.shape[0],gamma=cfg.DDPGAgent.gamma,tau=cfg.DDPGAgent.tau, epsilon=cfg.DDPGAgent.epsilon)
    # agent.setNoise(cfg.noise.sigma, cfg.noise.theta, cfg.noise.dt)
    for i in range(cfg.Training.epochs):
        obs, _ = env.reset()
        # gather experience
        agent.noise.reset()
        ep_actor_loss = 0
        ep_critic_loss = 0
        steps = 0
        for j in range(200):
            steps += 1
            env.render()
            action = agent.act(np.array([obs]), random_action=(i < 1)) # i < 1 weil bei ersten Epoche keine Policy vorhanden ist
            # execute action
            new_obs, r, done, _, _ = env.step(action)
            replay_buffer.put(obs, action, r, new_obs, done)
            obs = new_obs
            if done:
                break
                
        # Learn from the experiences in the replay buffer.
        for _ in range(128):
            s_states, s_actions, s_rewards, s_next_states, s_dones = replay_buffer.sample(cfg.Training.sample_size, cfg.Training.unbalance)
            actor_l, critic_l = agent.learn(s_states, s_actions, s_rewards, s_next_states, s_dones)
            ep_actor_loss += actor_l
            ep_critic_loss += critic_l
            
        if i % 25 == 0:
            avg_return = compute_avg_return(env, agent, num_episodes=2, render=False)
            print(
                f'epoch {i}, actor loss {ep_actor_loss / steps}, critic loss {ep_critic_loss / steps} , avg return {avg_return}')
        
    compute_avg_return(env, agent, num_episodes=10, render=True)
    env.close()

if __name__ == "__main__":
    main()