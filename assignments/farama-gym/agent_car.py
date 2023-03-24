import pickle, os

import gymnasium as gym
import numpy as np

from base_agent import BaseAgent, train, test

class TableCar(BaseAgent):
    def __init__(self, status:gym.spaces.Box, action:int=3,*,gamma:float=0.8, nlg_eps:int=2, will_load:bool=False):
        """mapping Box([-1.2 -0.07], [0.6 0.07]) to np.zeros([100, 100])
        unit = [0.018, 0.0014]
        (current - low)/unit
        e.g. array([-0.56397,  0.     ]) -> array([-31.33166667,   0.        ])
        """
        assert nlg_eps > 0, "nlg_eps means -lg(epsison) which define the precision of sampling"
        if will_load: return
        self._low = status.low
        self.gamma = gamma
        self.nlgeps = nlg_eps
        status_diff = status.high - status.low
        size = 10 ** nlg_eps
        self.unit = status_diff/size
        status_shape = [size] * status.shape[0]
        table_shape = status_shape + [action]
        self.table = np.zeros((table_shape))


    def _get_idx_list(self,status:np.ndarray)->np.ndarray:
        return np.array((status-self._low)/self.unit, dtype=np.uint16).tolist()

    def sample(self, status:np.ndarray)->int:
        idx = self._get_idx_list(status)
        return np.argmax(self.table[tuple(idx)])
    def update(self, status:tuple[np.ndarray, np.ndarray], action:int, reward:float, amp:float=1)->None:
        # actions = self._get_table_ref(status[0])
        # n_actions = self._get_table_ref(status[1])
        # actions[action] = reward + self.gamma * n_actions[action]
        last_status = self._get_idx_list(status[0])
        next_status = self._get_idx_list(status[1])
        next_action = self.sample(status[1])
        t0 = eval(f"self.table{last_status}")
        t1 = eval(f"self.table{next_status}")
        t0[action] = reward * amp + self.gamma * t1[next_action]
    def load(self,file:str)->bool:
        if not os.path.exists(file):
            return False
        with open(file, "rb") as fd:
            self.table, self.gamma, self.nlgeps, self._low, self.unit = pickle.load(fd)
        return True
    def save(self,file:str)->None:
        with open(file, "wb") as fd:
            pickle.dump((self.table, self.gamma, self.nlgeps, self._low, self.unit), fd)

def tt_env(fullname:str, short:str, *,load_iters:int=0, train_iters:int=4_000, nlg_epsion:int=2, seed:int=82, save_table:bool=True):
    train_env = gym.make(fullname)
    agent = TableCar(train_env.observation_space, train_env.action_space.n, nlg_eps=nlg_epsion, will_load=load_iters!=0)

    if load_iters != 0:
        agent.load(f"{short}-{load_iters}-{nlg_epsion}-{seed}.pkl")
    train(agent, train_env, train_iters, seed=seed)
    train_env.close()
    if save_table:
        agent.save(f"{short}-{load_iters + train_iters}-{nlg_epsion}-{seed}.pkl")

    test_env = gym.make(fullname, render_mode="human")
    print(test(agent, test_env, seed=seed))
    test_env.close()

if __name__=="__main__":
    tt_env("MountainCar-v0", "mc", train_iters=15_000, save_table=True)
    # tt_env("CartPole-v1", "cp", train_iters=5_000, nlg_epsion=1, seed=128) # poor performance