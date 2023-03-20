import gym

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
env.seed(42) # reward = -90.0
agent = BespokeAgent(env)

def play(env, agent, train:bool=False):
    episode_reward = 0
    ob = env.reset()
    while True:
        env.render()
        action = agent.decide(ob)
        next, reward, term, _ = env.step(action)
        episode_reward += reward
        if train:
            agent.learn()
        if term:
            break
        ob = next
    return episode_reward

reward = play(env, agent)
print(f"reward = {reward}")
env.close()
