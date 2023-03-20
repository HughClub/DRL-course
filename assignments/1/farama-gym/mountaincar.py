import gymnasium as gym

class BespokeAgent:
    def __init__(self, env: gym.Env):
        super().__init__()
    def decide(self, obeservation)->int:
        if isinstance(obeservation, tuple):
            position, velocity = obeservation[0]
        else:
            position, velocity = obeservation
        lb = min(-0.09 * (position + 0.25) ** 2 + 0.03, 0.3 * (position + 0.9) ** 4 - 0.008)
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action # 返回动作
    def learn(self, *args):
        pass
env = gym.make("MountainCar-v0", render_mode="human")
agent = BespokeAgent(env)

def play(env:gym.Env, agent, seed:int, train:bool=False):
    episode_reward = 0
    ob = env.reset(seed=seed)
    while True:
        env.render()
        action = agent.decide(ob)
        next, reward, term, trunc, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn()
        if term or trunc:
            break
        ob = next
    return episode_reward



reward = play(env, agent, 82) # reward=-83
print(f"{reward=}")
env.close()


# get the max reward test seed in [0, 1024)
def test_env():
    env = gym.make("MountainCar-v0")
    agent = BespokeAgent(env)

    def iplay(env:gym.Env, agent, seed:int, train:bool=False):
        episode_reward = 0
        ob = env.reset(seed=seed)
        while True:
            action = agent.decide(ob)
            next, reward, term, trunc, _ = env.step(action)
            episode_reward += reward
            if train:
                agent.learn()
            if term or trunc:
                break
            ob = next
        return episode_reward

    turn, maxr = 0, -200
    for i in range(1024):
        r = iplay(env, agent, i)
        if r > maxr:
            maxr, turn = r, i
    print(maxr, turn)

# test_env() # -83.0 82