# env: https://gym.openai.com/envs/Tennis-ram-v0/

import gym
import numpy as np

num_episodes = 1

env = gym.make("ALE/MsPacman-ram-v5")

print(gym.__version__)
# Discrete(9)
print(env.action_space)
# Box([...][...], (128,), uint8)
print(env.observation_space)

for _ in range(num_episodes):
    obs = env.reset()
    while True:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)

        if done:
            # print(obs, reward, action)
            break

    # Anything to record after 1 episode?