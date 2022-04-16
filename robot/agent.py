import copy
import numpy as np

class CliffWalk():
    def __init__(self, start, end, states, q_fct,
                 actions, rewards, alpha, e):
        self.start = start
        self.end = end
        self.states = states
        self.initial_q_fct = q_fct
        self.q_fct = copy.deepcopy(self.initial_q_fct)
        self.actions = actions
        self.rewards = rewards

        self.e = e  # epsilon
        self.alpha = alpha  # lr

        self.sum_rewards = []

    def reset(self):
        # reset sum of rewards for measuring performance during training
        self.sum_rewards = []

        # initialize q_fct
        self.q_fct = copy.deepcopy(self.initial_q_fct)

    def reward_fct(self, current_state, action, rewards):
        next_state = self.act(current_state, action)
        return rewards[next_state]

    def transition(self, current_state, action):
        i, j = current_state
        if action == "U":
            return (i - 1, j)
        elif action == "D":
            return (i + 1, j)
        elif action == "L":
            return (i, j - 1)
        return (i, j + 1)

    def act(self, state, actions):
        p = np.random.random()

        # e-greedy action
        if p < self.e:
            return np.random.choice(actions[state], 1)[0]
        else:
            return max(self.q_fct[state], key=self.q_fct[state].get)

    def sarsa(self, num_of_episodes):
        for _ in range(num_of_episodes):
            current_state = self.start
            sum_reward = 0
            while True:
                # choose A from S using policy derived from e-greedy
                action = self.act(current_state, actions)

                # take A, observe R, S'
                next_state = self.transition(current_state, action)
                reward = self.rewards[next_state]

                # choose A' from S' using policy derived from e-greedy
                next_action = self.act(next_state, actions)

                # Update: q(s,a) <- q(s,a) + alpha * (r + q(s',a') - q(s,a))
                self.q_fct[current_state][action] += self.alpha * (
                        reward + self.q_fct[next_state][next_action] - self.q_fct[current_state][action]
                )

                sum_reward += reward

                # check if S' is terminal
                if reward == -100 or next_state == self.end:
                    self.sum_rewards.append(sum_reward)
                    break

                # S <- S'
                current_state = next_state

    def qlearning(self, num_of_episodes):
        for _ in range(num_of_episodes):
            current_state = self.start
            sum_reward = 0
            while True:
                # choose A from S using policy derived from e-greedy
                action = self.act(current_state, actions)

                # take A, observe R, S'
                next_state = self.transition(current_state, action)
                reward = self.rewards[next_state]

                # Q-update by maximizing over Q(s',a)
                self.q_fct[current_state][action] += self.alpha * (
                        reward + max(self.q_fct[next_state].values()) - self.q_fct[current_state][action]
                )

                sum_reward += reward

                # check if S' is terminal
                if reward == -100 or next_state == self.end:
                    self.sum_rewards.append(sum_reward)
                    break

                # S <- S'
                current_state = next_state