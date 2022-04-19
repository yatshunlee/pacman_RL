import os, copy
import gym
import torch
import torch.nn as nn
import numpy as np
# from ddqn_train import DDQN

class DDQN(nn.Module):
    """
    The Double Deep Q-Network has as input a state s and
    outputs the state-action values q(s,a_1), ..., q(s,a_n) for all n actions.
    :param: state_dim: for input layer
    :param: hidden_dim: for every hidden layer
    :param: action_dim: for output layer
    """
    def __init__(self, action_dim, state_dim, hidden_dim):
        super().__init__()

        self.online = nn.Sequential(
            nn.Linear(state_dim, hidden_dim*2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen.
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, input, model):
        """
        When doing update by forward, it takes:
        :param: input: all state of each observation
        :param: model: online or target
        :return: Q_values of all actions given state from online/target
        """

        if model == "online":
            return self.online(input)
        elif model == "target":
            return self.target(input)


if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    PATH = 'checkpoints/2022-04-18T23-12-00/pacman_ddqn_44.chkpt' #'data/expert_CE.chkpt' #'data/expert_CE.chkpt' #
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