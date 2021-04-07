# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os

import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim

from arguments import parse_args
from replay_buffer import ReplayBuffer
import multiagent.scenarios as scenarios
from model import openai_actor, openai_critic
# from multiagent.environment import MultiAgentEnv
from multiagent.environment_uav import MultiAgentEnv


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


def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist):
    """
    init the trainers or load the old model
    """
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(env.n):
        if arglist.restore:
            actors_cur[i] = torch.load(arglist.old_model_name + 'a_c_{}.pt'.format(i))
            actors_tar[i] = torch.load(arglist.old_model_name + 'a_t_{}.pt'.format(i))
            critics_cur[i] = torch.load(arglist.old_model_name + 'c_c_{}.pt'.format(i))
            critics_tar[i] = torch.load(arglist.old_model_name + 'c_t_{}.pt'.format(i))
        else:
            actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
            actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
            critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)

        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur
    return actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c


def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key] * tao + \
                                (1 - tao) * state_dict_t[key]
        agent_t.load_state_dict(state_dict_t)
    return agents_tar


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                 actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c):
    """ 
    use this func to make the "main" func clean
    par:
    |input: the data for training
    |output: the data for next update
    """
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
            (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0: print('\r=start training ...' + ' ' * 100)
        # update the target par using the cur
        update_cnt += 1

        # update every agent in different memory batch
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None: continue  # jump to the next model update

            # sample the experience
            _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)  # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                    for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q
            q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_
            tar_value = q_ * arglist.gamma * done_n + rew  # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value)  # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()

            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the aciton of this agent
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-3 * loss_pse + loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao)
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)

    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar


def log_print(env):
    print('*********************Bandwidht Assignment**********************')
    for i in env.world.bandwidth:
        for j in i:
            print('%10.2e' % j, end='')
        print('\n', end='')

    print('*****************************SNR*******************************')
    for i in env.world.SNR:
        for j in i:
            print('%10.2f' % j, end='')
        print('\n', end='')
    print('**************************Throughput***************************')
    for i in env.world.Rate:
        for j in i:
            print('%10.2e' % j, end='')
        print('\n', end=' ')
    print('\n')


def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    # print(env.action_space[0].high)
    # action_shape_n = [sum(env.action_space[i].shape[0]) for i in range(env.n)]  # no need for stop bit
    action_shape_n = [sum(env.action_space[i].high + 1) for i in range(env.n)]  # no need for stop bit
    # print(action_shape_n)
    num_adversaries = min(env.n, arglist.num_adversaries)
    actors_cur, critics_cur, actors_tar, critics_tar, optimizers_a, optimizers_c = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, arglist)

    # memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    update_cnt = 0
    max_ep_r = None
    mean_ep_r_list = []
    me_ag_r_list = [[] for _ in range(env.n)]
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)]  # set the init reward
    agent_info = [[[]]]  # placeholder for benchmarking info
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
        # print('obs',obs_size)
        # print('act',action_size)

    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()

    for episode_cnt in range(arglist.max_episode):

        for episode_step in range(arglist.per_episode_max_len):
            # get action
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                        for agent, obs in zip(actors_cur, obs_n)]
            # print('action', action_n)

            # interact with env
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)

            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n): agent_rewards[i][-1] += rew

            game_step += 1

            if not arglist.test:
                # save the experience
                memory.add(obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n)
                # train our agents
                update_cnt, actors_cur, actors_tar, critics_cur, critics_tar = agents_train( \
                    arglist, game_step, update_cnt, memory, obs_size, action_size, \
                    actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)

            # update the obs_n
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len - 1)
            if done or terminal:
                continue

            if arglist.display:
                time.sleep(0.1)
                env.render()
                log_print(env)


        obs_n = env.reset()
        agent_info.append([[]])
        episode_rewards.append(0)
        for a_r in agent_rewards:
            a_r.append(0)

        # cal the reward print the debug data
        if (episode_cnt+1) % arglist.fre4save_model == 0:
            print('######################## Episodes: {}  Steps: {} #############################'.format(episode_cnt+1, game_step))
            mean_agents_r = [round(np.mean(agent_rewards[idx][-arglist.fre4save_model:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-arglist.fre4save_model:-1]), 3)

            time_now = time.strftime('%y_%m%d_%H%M')
            if max_ep_r is None or mean_ep_r > max_ep_r:
                max_ep_r = mean_ep_r

                model_file_dir = os.path.join(arglist.save_dir, '{}/{}_{}/'.format(
                    arglist.exp_name, time_now, game_step))
                if not os.path.exists(model_file_dir):  # make the path
                    os.makedirs(model_file_dir)
                for agent_idx, (a_c, a_t, c_c, c_t) in \
                        enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                    torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                    torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                    torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                    torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

            print('time:{} step: {}  episodes: {}  mean episode reward: {}  time cost: {}'.format(time_now, game_step,
                                                                                                   episode_cnt+1,
                                                                                                   mean_ep_r,
                                                                                                   round(time.time() - t_start, 3)))
            t_start = time.time()
            log_print(env)  # print log information to keep track of training process

            # record mean reward for plotting learning curves
            mean_ep_r_list.append(mean_ep_r)
            for i, mean_ag_r in enumerate(mean_agents_r):
                me_ag_r_list[i].append(mean_ag_r)

            plots_dir = arglist.plots_dir + arglist.exp_name
            if not os.path.exists(plots_dir): os.makedirs(plots_dir)
            rew_file_dir = plots_dir + '/rewards.pkl'
            with open(rew_file_dir, 'wb') as fp:
                pickle.dump(mean_ep_r_list, fp)
            agrew_file_dir = plots_dir + '/agrewards.pkl'
            with open(agrew_file_dir, 'wb') as fp:
                pickle.dump(me_ag_r_list, fp)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
