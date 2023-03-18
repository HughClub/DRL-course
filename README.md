# Deep Reinforcement Learning

深度强化学习课程作业

推荐的课外资料 EasyRL
- online版本: <https://datawhalechina.github.io/easy-rl/>
- 代码: <https://github.com/datawhalechina/easy-rl>

## 环境创建

使用 conda:

```bash
conda env create --file env.gym.yml # 使用 Python3.7+OpenAI-Gym
conda env create --file env.yml # 使用 Python3.10+gymnasium
```

或者使用 pip

```bash
pip install gym==0.25.2 pygame==2.1.0 # gym
pip install gymnasium gymnasium[classic_control]  # gymnasium

```

> gymnasium 是 OpenAI 不管 gym 之后的官方社区包

## 课堂展示内容的源码

Q-Learning:
- 来自于: [EasyRL QLearning](https://github.com/datawhalechina/easy-rl/blob/master/notebooks/Q-learning/QLearning.ipynb)


DQN:
- 发的FlappyBird的仓库: <https://github.com/SukerZ/Playing-Flappy-Bird-by-DQN-on-PyTorch>
- 展示的内容来自: <https://github.com/yenchenlin/DeepLearningFlappyBird>