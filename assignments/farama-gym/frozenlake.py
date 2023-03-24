import gymnasium as gym

from cliffwalking import main

def cheater():
    """action
    - 0: Move left
    - 1: Move down
    - 2: Move right
    - 3: Move up
    """
    env:gym.Env = gym.make("FrozenLake-v1", render_mode="human",is_slippery=False)
    action_list = [1,1,2,2,1,2]
    rewards = 0
    env.reset()
    for act in action_list:
        env.render()
        _, reward, term, _, _ = env.step(act)
        rewards += reward # type: ignore
        if term:
            break
    env.close()
    return rewards

if __name__=="__main__":
    main("FrozenLake-v1",round=1_000,is_slippery=False)
    # print(cheater())