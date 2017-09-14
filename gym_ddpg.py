from ddpg import *
import gym
import logging
import argparse

ENV_NAME = 'MountainCarContinuous-v0'#'Pendulum-v0'#
PATH = 'models/'
EPISODES = 100000
TEST = 1

def main():
    env = gym.make(ENV_NAME)
    env.reset()
    agent = DDPG(env)

    returns = []
    rewards = []

    rs = RunningStats()
    s,s1=None,None
    for episode in xrange(EPISODES):
        s = env.reset()
        reward_episode = []
        print "episode:",episode
        # Train
        n_step = 1 # n_step return
        s = rs.normalize(s)
        for step in xrange(1000):
            ac = agent.action(s)+int(not agent.start_train)*agent.exploration_noise.noise()
            #print(ac)
            temp = 0
            env.render()
            for i in range(n_step):
                ob, rew, new, _ = env.step(ac)
                temp += rew * GAMMA**i
                if new: 
                    break
            rew = temp
            #if rew > 80.:
                #print(rew)
            s1 = rs.normalize(ob)
            agent.perceive(s,ac,rew,s1,new)
            s = s1
            reward_episode.append(rew)
            if new:
                break


        if episode % 1 == 0:
            print("episode reward = %.2f" % sum(reward_episode))
        # Testing:
        #if episode % 1 == 0:
        if episode % 10 == 0 and episode > 50:
            #agent.save_model(PATH, episode)

            total_return = 0
            ave_reward = 0
            for i in xrange(TEST):
                s = env.reset()
                reward_per_step = 0
                s = rs.normalize(s)
                for j in xrange(1000):
                    ac = agent.action(s)
                    temp = 0
                    for i in range(n_step):
                        ob, rew, new, _ = env.step(ac)
                        temp += rew
                        if new: 
                            break
                    rew = temp
                    s1 = rs.normalize(s)
                    s = s1
                    total_return += rew
                    if new:
                        break
                    reward_per_step += (rew - reward_per_step)/(j+1)
                ave_reward += reward_per_step

            ave_return = total_return/TEST
            ave_reward = ave_reward/TEST
            returns.append(ave_return)
            rewards.append(ave_reward)

            print 'episode: ',episode,'Evaluation Average Return:',ave_return, '  Evaluation Average Reward: ', ave_reward

if __name__ == '__main__':
    main()
