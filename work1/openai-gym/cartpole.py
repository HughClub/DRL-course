import gym

def cartpolev1():
    env:gym.Env = gym.make("CartPole-v1", render_mode="human")
    env.reset()

    max_iter, start_iter = 0, 0
    for iter in range(100):
        env.render()
        action = env.action_space.sample()
        ob, reward, term, info = env.step(action)
        if term:
            this_time = iter - start_iter
            if this_time > max_iter:
                max_iter = this_time
                print(f"{start_iter}-{iter} \t max_iter:{max_iter}")
                start_iter = iter+1
            env.reset()
    env.close()

if __name__=="__main__":
    cartpolev1()