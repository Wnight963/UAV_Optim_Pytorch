import os
import sys

import torch
import multiagent.scenarios as scenarios
from multiagent.environment_uav import MultiAgentEnv

from model import actor_agent, critic_agent
from arguments import parse_args

def make_env(scenario_name, arglist, benchmark=False):
    """ 
    create the environment from script 
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, arglist):
    trainers_cur = []
    trainers_tar = []
    optimizers = []
    input_size = [8, 10, 10] # the obs size
    input_size_global = [23, 25, 25] # cal by README

    """ load the model """
    actors_tar = [torch.load(arglist.old_model_name+'a_t_{}.pt'.format(agent_idx), map_location=arglist.device) \
        for agent_idx in range(env.n)]

    return actors_tar

def enjoy(arglist):
    """ 
    This func is used for testing the model
    """

    episode_step = 0
    """ init the env """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    """ init the agents """
    obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
    actors_tar = get_trainers(env, arglist)

    """ interact with the env """
    obs_n = env.reset()
    while(True):

        # update the episode step number
        episode_step += 1
        # print(episode_step)

        # get action
        try:
            action_n = []
            # action_n = [agent.actor(torch.from_numpy(obs).to(arglist.device, torch.float)).numpy() \
            # for agent, obs in zip(trainers_cur, obs_n)]
            for actor, obs in zip(actors_tar, obs_n):
                action = torch.clamp(actor(torch.from_numpy(obs).to(arglist.device, torch.float)), -1, 1)
                action_n.append(action)


        except:
            print(obs_n)

        # interact with env
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)

        print('*********************Bandwidht assignment**********************')
        for i in env.world.bandwidth:
            for j in i:
                print('%10.2e' % j, end='')
            print('\n', end='')

        print('*************************SNR*****************************')
        for i in env.world.SNR:
            for j in i:
                print('%10.2f' % j, end='')
            print('\n', end='')
        print('*************************Rate*****************************')
        for i in env.world.Rate:
            for j in i:
                print('%10.2e' % j, end='')
            print('\n', end=' ')
        print('\n')

        # update the flag
        done = all(done_n)
        terminal = (episode_step >= arglist.per_episode_max_len)

        # reset the env
        if done or terminal: 
            episode_step = 0
            obs_n = env.reset()

        # render the env
        # print(rew_n)
        env.render()

if __name__ == '__main__':
    arglist = parse_args()
    enjoy(arglist)
