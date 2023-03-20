import pickle, os

import gymnasium as gym
import numpy as np

from base_agent import BaseAgent, train, test

def cheater():
    """action
    0, 1, 2, 3: up right down left
    """
    env:gym.Env = gym.make("CliffWalking-v0", render_mode="human")
    action_list = [0] + [1]*11 + [2]
    rewards = 0
    env.reset()
    for act in action_list:
        env.render()
        _, reward, term, _, _ = env.step(act)
        rewards += reward # type: ignore
        # print(f"{term=}")
    env.close()
    return rewards

class Walker(BaseAgent):
    def __init__(self, n_status:int, n_action:int,*, gamma:float=0.8):
        super().__init__()
        self.table = np.zeros((n_status, n_action)) # status * action
        self.gamma = gamma
    def sample(self, status:int)->int:
        return np.argmax(self.table[status])
    def update(self, status:tuple[int, int], action:int, reward:float)->None:
        next_action = self.sample(status[1])
        self.table[status[0], action] = reward + self.gamma * self.table[status[1], next_action]
    def load(self,file:str)->bool:
        if not os.path.exists(file):
            return False
        with open(file, "rb") as fd:
            self.table, self.gamma = pickle.load(fd)
        return True
    def save(self,file:str)->None:
        with open(file, "wb") as fd:
            pickle.dump((self.table, self.gamma), fd)

def main(round:int=32):
    train_env = gym.make("CliffWalking-v0")
    agent = Walker(train_env.observation_space.n, train_env.action_space.n)
    train(agent, train_env, round, interval=4)
    train_env.close()

    test_env = gym.make("CliffWalking-v0", render_mode="human")
    print("test reward:", test(agent, test_env))
    test_env.close()


if __name__ == "__main__":
    # print(cheater())
    main(36)