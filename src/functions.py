from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np

def compute_avg_return(env, agent, num_episodes=1, max_steps=200, render=False):
    total_return = 0.0
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_return = 0.0
        done = False
        steps = 0
        while not (done or steps > max_steps):
            if render:
                clear_output(wait=True)
                plt.axis('off')
                plt.imshow(env.render())
                plt.show()
            action = agent.act(np.array([obs]))
            obs, r, done, _, _ = env.step(action)
            episode_return += r
            steps += 1
        total_return += episode_return
    return total_return / num_episodes