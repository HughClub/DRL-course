from gymnasium import Env

class BaseAgent:
    def sample(self, status)->int: raise NotImplementedError
    def update(self, status, action, reward)->None: raise NotImplementedError
    def load(self, file:str)->bool: raise NotImplementedError
    def save(self, file:str)->None: raise NotImplementedError

def train(agent: BaseAgent, env:Env, iters:int, seed:int=42, *,
          interval:int=200, verbose:bool=True):
    for iter in range(iters):
        round_reward:float = 0.0
        status, _ = env.reset(seed=seed)
        while True:
            action = agent.sample(status)
            next, reward, term, trunc, _ = env.step(action)
            agent.update((status, next), action, reward)
            status = next
            round_reward += reward # type: ignore
            if term or trunc:
                if verbose:
                    if iter % interval == 0:
                        print(f"round: {iter}\t reward: {round_reward}")
                    if iter == iters-1:
                        print(f"round: {iter}\t reward: {round_reward}")
                break

def test(agent: BaseAgent, env:Env, seed:int=42, render:bool=True):
    round_reward:float = 0.0
    status, _ = env.reset(seed=seed)
    while True:
        if render:
            env.render()
        action = agent.sample(status)
        next, reward, term, trunc, _ = env.step(action)
        status = next
        round_reward += reward # type: ignore
        if term or trunc:
            return round_reward