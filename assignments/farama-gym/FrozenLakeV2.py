from typing import Optional, Any

import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv

"""
Just a RewardWappered FrozenLake
reference: https://gymnasium.farama.org/environments/toy_text/frozen_lake/#information
"""
class FrozenLakeWrapper:
    """
    all things are equal to FrozenLake-v1 except Reward
    Reward schedule:
    - Reach goal: +10
    - Reach hole: -10
    - Reach frozen: -1
    """
    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }
    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        self._env:FrozenLakeEnv = gym.make("FrozenLake-v1", render_mode=render_mode, desc=desc,
                             map_name=map_name, is_slippery=is_slippery)
        self.desc = "".join([
            i.decode()
            for line in self._env.desc
            for i in line
        ])
        self._reward_map = {
            'S': 0,
            'F': -1,
            'H': -10,
            'G': 10
        }
    def __getattr__(self, name: str) -> Any:
        """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
        if name == "_np_random":
            raise AttributeError(
                "Can't access `_np_random` of a wrapper, use `self.unwrapped._np_random` or `self.np_random`."
            )
        elif name.startswith("_"):
            raise AttributeError(f"accessing private attribute '{name}' is prohibited")
        return getattr(self._env, name)
    @property
    def unwrapped(self)->gym.Env:
        return self._env.unwrapped
    def step(
        self, action: int
    ) -> tuple[int, float, bool, bool, dict[str, Any]]:
        """Uses the :meth:`step` of the :attr:`env` that can be overwritten to change the returned data."""
        obs, _, term, trunc, info = self._env.step(action)
        reward = self._reward_map[self.desc[obs]]
        return obs, reward, term, trunc, info
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[int, dict[str, int]]:
        """Uses the :meth:`reset` of the :attr:`env` that can be overwritten to change the returned data."""
        return self._env.reset(seed=seed, options=options)

    def render(self):
        """Uses the :meth:`render` of the :attr:`env` that can be overwritten to change the returned data."""
        return self._env.render()

    def close(self):
        """Closes the wrapper and :attr:`env`."""
        return self._env.close()


def make(**kwargs)->FrozenLakeWrapper:
    return FrozenLakeWrapper(**kwargs)

from cliffwalking import Walker, train, test
def main(round:int=100, seed:int=42, interactive:bool=True)->float:
    train_env = make()
    agent = Walker(train_env.observation_space.n, train_env.action_space.n)
    train(agent, train_env, round, seed, verbose=interactive)
    train_env.close()

    test_env = make(render_mode="human") if interactive else train_env
    tre = test(agent, test_env, seed, interactive)
    test_env.close()
    return tre

if __name__=="__main__":
    print(main(round=30))