import matplotlib.pyplot as plt
import numpy as np

def compute_avg_return(env, agent, num_episodes=1, max_steps=200, render=False):
    total_return = 0.0
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        steps = 0
        while not (done or steps > max_steps):
            if render:
                env.render()
            action = agent.act(np.array([obs]),explore=False)
            obs, r, done, _, _ = env.step(action)
            episode_return += r
            steps += 1
        returns.append(episode_return)
        total_return += episode_return
    return total_return / num_episodes, np.std(returns)