import os
import gym
import torch
import numpy as np
from ddqn_train import DDQN

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    PATH = 'expert.chkpt'
    env = gym.make("ALE/MsPacman-ram-v5", render_mode='human') #
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hidden_dim = 128

    model = DDQN(action_dim=env.action_space.n,
                 state_dim=env.observation_space.shape[0],
                 hidden_dim=hidden_dim).to(device)
    print(model)
    # torch.load() -> dict and load_state_dict into the model
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model'])

    for i_episode in range(5):
        obs = env.reset()

        while True:
            # env.render()
            obs = torch.Tensor(obs).to(device)

            with torch.no_grad():
                values = model(obs, "online")

            action = np.argmax(values.cpu().numpy())
            obs, reward, done, info = env.step(action)

            if done:
                break

    env.close()