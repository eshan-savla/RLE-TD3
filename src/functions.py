import matplotlib.pyplot as plt
import numpy as np

def compute_avg_return(env, agent, num_episodes=1, max_steps=200, render=False):
    total_return = 0.0
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated or steps > max_steps):
            if render:
                env.render()
            action = agent.act(np.array([obs]),explore=False)
            obs, r, terminated, truncated, info, _ = env.step(action)
            done = terminated or truncated
            episode_truncated = not done or info.get("TimeLimit.truncated", False)
            info["TimeLimit.truncated"] = episode_truncated
            # truncated may have been set by the env too
            truncated = truncated or episode_truncated
            episode_return += r
            steps += 1
        returns.append(episode_return)
        total_return += episode_return
    return total_return / num_episodes, np.std(returns)