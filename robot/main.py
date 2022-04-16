import gym
env = gym.make('FetchSlide-v1') #CartPole-v0
observation = env.reset()

print(env.action_space)
# for i_episode in range(20):
#     observation = env.reset()
#     for t in range(100):
#         env.render(False)
#         action = env.action_space.sample() # your agent here (this takes random actions)
#         observation, reward, done, info = env.step(action)
#
#         if done:
#             print("Episode finished after {} timesteps".format(t + 1))
#             break
#
env.close()