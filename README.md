# Using DDQN, PER, Imitation Learning to Train an Pacman Agent
### Environment
<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167265376-562a4978-89d8-40e3-bf70-a14a9ca1baf4.png" alt=""/>
</p>

The gym environment setting is listed below:
- The environment name: “MsPacman-ram-v5”
- Observation space: RAM content with shape (128,)
- Action space: Discrete(9). U – Up, R – Right, L – Left, D – Down

    |   0	  |  1	|  2	|  3	|  4	|  5	|  6	|  7	|  8  |
    | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
    |  STAY |  U  |  R  |  L  |  D  |  UR |  UL |  DR |  DL |

The default reward design:
- 10 points for each small pellet
- 50 points for each big pellet
- 100 points for each cherry
- (200 * 1 + ... + 200 * N) points for killing N ghost(s) during invincible mode


### Double Deep Q Network (DDQN)
The deep Q network method is a value function approximation technique to approximate the Q values by deep learning.
It avoids the trouble of maintaining a Q table and only estimates the Q values by inputting the observation state.
To address the induced overestimation issue of DQN, Double Deep Q Network uses an online network and a target network to estimate the TD target.
<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167265329-e6f5a65d-d442-4696-9e1f-5837e781e5ca.png" alt=""/>
</p>
I explained the details in QWOP_RL: https://github.com/yatshunlee/qwop_RL. If want to know more, please take a look on that project.

### Prioritized Experience Replay (PER)
Instead of using experience replay, providing no chance for the agent to select some relatively fruitful and rare experience, the agent was then trained with prioritized experience replay by DDQN. In PER, we want to prioritize the agent to learn from the experience of a big difference between TD estimate and TD target.
That means the agent can learn a lot from the big difference to minimize the loss.
The implementation is to assign a learning value to each experience tuple in the replay buffer.
It indicates the priority of each sample experience which is directly proportional to the absolute value of the TD error with a small constant offset.

Illustration of the Prioritized Replay Buffer, where (exp_k, p_k) stores the kth experience and priority value.

<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167265078-0569b23c-204a-41d0-99fa-f82ec07c6ee0.png" alt=""/>
</p>

When sampling from the buffer, we can convert the value into the probability of choosing that tuple and
add a scaling constant power a∈[0,1] to each P_i since the priority value is from the unstable TD error.

<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167264807-e2a51aca-640d-4cc2-80af-69bc0977a600.png" alt=""/>
</p>

Since sampling with priorities will create bias in the neural network towards the experiences with a higher priority and overfit them, we have to address this issue by adding a normalized weight to the weight update process.

<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167264961-b3279ef8-5296-4219-a3c6-d0c92588043c.png" alt=""/>
</p>

This avoids excessive update steps from the increased training frequency of those experiences. The bias correction is applied by b, starting from a low value and increasing to 1 overtime.

### Imitation Learning
Rather than shaping the reward, I decided to implement one of the imitation learning methods: the behavioral cloning method.
It’s a simple but robust technique to enhance the agent's performance.
The idea is to pre-train the neural network with an expert dataset.

Since most of the problems in the world do not have an explicit reward, by behavioral cloning, the agent can learn without a reward.
Since the observation data is a RAM content, it is hard for a human to understand the content and perform reward shaping.
Hand-crafted rewards may also lead to uncontrolled behavior.
With imitation learning, we can directly learn from the expert data.

Illustration of the Behavioral Cloning Method
<p align="center">
  <img src="https://user-images.githubusercontent.com/69416199/167265249-78a89d72-fb36-4808-ac7d-17cb0bdd1ec6.png" alt=""/>
</p>

### Code
to record gaming experience and run the notebook `imitation_learning.ipynb` to perform imitation learning
    
    python expert_recorder.py
to train by DDQN

    python ddqn_train.py
to test by DDQN

    python ddqn_test.py
