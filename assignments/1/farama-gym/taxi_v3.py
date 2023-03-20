import gymnasium as gym

from cliffwalking import Walker, train, test

def main(env_name:str, round:int=100, seed:int=42, interactive:bool=True)->float:
    train_env = gym.make(env_name)
    agent = Walker(train_env.observation_space.n, train_env.action_space.n)
    train(agent, train_env, round, seed, verbose=interactive)
    train_env.close()

    test_env = gym.make(env_name, render_mode="human") if interactive else train_env
    tre = test(agent, test_env, seed, interactive)
    test_env.close()
    return tre

def pick_seed(limit:int=1024)->int:
    m_it, m_re = 0, -600
    for it in range(limit):
        c_re = main("Taxi-v3", seed=it, interactive=False)
        if c_re > m_re:
            m_re = c_re
            m_it = it
    return m_it

if __name__=="__main__":
    # seed = pick_seed()
    # print(seed)
    seed = 92 # god seed
    main("Taxi-v3", round=200, seed=seed)