import numpy as np
from multiagent.core_uav import World, Agent, Landmark
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 10
        world.dim_pow = 10
        num_uav = 3
        world.collaborative = True  # whether agents share rewards
        # add agents
        world.agents = [Agent() for i in range(num_uav)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(world.dim_c)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = True

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # set color for landmarks
        # world.landmarks[0].color = np.array([0.25, 0.25, 0.75])
        # world.landmarks[1].color = np.array([0.25, 0.25, 0.75])
        # world.landmarks[2].color = np.array([0.25, 0.25, 0.75])
        # world.landmarks[3].color = np.array([0.25, 0.25, 0.75])
        # world.landmarks[4].color = np.array([0.25, 0.25, 0.75])

        # set random initial states
        # for i, agent in enumerate(world.agents):
        #     agent.color = np.array([0.35, 0.35, 0.85])
        #     # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.25, 0.25, 0.25])
            # set random initial states
        for i, agent in enumerate(world.agents):
            agent.color = np.array([[0.35, 0.35, 0.85], [0.25, 0.75, 0.25], [0.5, 0.15, 0.35]])[i]
            # agent.state.p_pos = np.array([[0.35, 0.35], [-0.35, -0.35]])[i]
            agent.state.p_pos = np.array([[0., 0.5], [-0.43, -0.25], [0.43, -0.25]])[i]
            # agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

            agent.state.assign = np.zeros(world.dim_c)
            agent.state.SNRforUser = np.zeros(world.dim_c)
            agent.state.assign_pow = np.zeros(world.dim_c)

        for i, landmark in enumerate(world.landmarks):
            # landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            # landmark.state.p_pos = np.array([[-0.8154612  , -0.24502182], [0.89699076, -0.14581629],
            #                                  [-0.324932, 0.60295496], [0.62360916, -0.52245444],
            #                                  [0.65862443, 0.23426809]])[i]

            landmark.state.p_pos = np.array([[-0.8154612, -0.24502182], [0.89699076, -0.14581629],
                                             [-0.3249325, 0.60295496], [0.62360916, -0.52245444],
                                             [0.65862443, 0.23426809], [0.29590076, 0.54571629],
                                             [-0.89699076, -0.74682629], [0.84629571, -0.98582649],
                                             [-0.89699076, 0.14581629], [-0.14699076, -0.75584629]])[i]

            landmark.state.p_vel = np.zeros(world.dim_p)
            # landmark.state.p_vel = np.random.uniform(-1, +1, world.dim_p)
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.assign_index = None
            landmark.switch_cout = 0

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return rew, collisions, min_dists, occupied_landmarks

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        try:
            min_rate = None
            for i in range(world.dim_c):
                min_rate = np.max(world.Rate[:, i]) if min_rate is None else min(min_rate, np.max(world.Rate[:, i]))
            reward = 10*np.log(min_rate+1e-9)
            # reward = 10*np.log(np.min(world.Rate[np.nonzero(world.Rate)]))
            # reward = np.log(np.min(agent.state.UAV_Rate[np.nonzero(agent.state.UAV_Rate)]))
            # reward = np.log(np.sum(world.Rate))
        except ValueError:
            reward = 0

        # reward = np.log(np.sum(world.Rate))

        for i in range(world.dim_c):
            if np.max(world.SNR[:, i]) < 15:
                reward -= 10
                #print('SNR not saftistied')

        for agent in world.agents:
            if not (-1 <= agent.state.p_pos[0] <= 1 and -1 <= agent.state.p_pos[1] <= 1):
                reward -= 100
                # print(entity.name)
                # print(entity.state.p_pos)
                # raise Exception('超出边界')

        # 用户的切换次数作为负奖励
        for landmark in world.landmarks:
            reward -= landmark.switch_cout
            landmark.switch_cout = 0

        if agent.collide:
            for a in world.agents:
                if a.name is not agent.name:
                    if self.is_collision(a, agent):
                        reward -= 100

        return reward

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        assign = []
        SNRforUser = []
        assign_pow = []
        for other in world.agents:
            assign.append(other.state.assign)
            SNRforUser.append(other.state.SNRforUser)
            assign_pow.append(other.state.assign_pow)
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        # print(assign)
        # print(assign_pow)
        # return np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

        # return np.concatenate(
        #     [agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + assign)

        maxSNR = np.max(SNRforUser)
        normalized_SNR = SNRforUser.copy()
        for i, item in enumerate(SNRforUser):
            if maxSNR != 0: normalized_SNR[i] = item / maxSNR

        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + assign + normalized_SNR+ assign_pow)
