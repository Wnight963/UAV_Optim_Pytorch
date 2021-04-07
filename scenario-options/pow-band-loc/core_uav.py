import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None
        self.hold_action = 0

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication allocation
        self.c = None
        self.UAV_Rate = None
        self.SNRforUser = None
        self.assign = None

        self.assign_pow = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

        self.pow = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name
        self.name = ''
        # properties:
        self.size = 0.0250
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = 10
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.assign_index = None
        self.switch_cout = 0

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2

        self.dim_pow = 0

        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3
        self.collaborative = False

        self.pow_UAVs = 1000  # 无人机发射功率
        self.pow_GBS = None  # 地面基站发射功率
        self.gain_UAV = 10**(-40/10)  # 单位db 无人机链路单位距离下的信道增益
        self.gain_GBS = 60  # 单位db 地面基站链路单位距离下的信道增益
        self.los_exp_UAV = 2  # 无人机链路衰减指数
        self.los_exp_GBS = 3  # 地面基站链路衰减指数
        self.max_band = 10e6  # 最大可用带宽
        self.noise_pdf = 10**(-167/10)  # 噪声功率谱密度 dBm

        self.bandwidth = None  # 带宽
        self.tx_pow = None  # 发射功率
        self.Rate = None
        self.SNR = None

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self):
        # set actions for scripted agents
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        # p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)

        self.update_landmark_pos()

        # self.update_agent_pos()

        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)
        self.update_world_state()

        # dict_state = []
        # for i, landmark in enumerate(self.landmarks):
        #     # Message = 'Users {} locats in {}'.format(i, landmark.state.p_pos)
        #     dict_state.append({'User {}'.format(i): landmark.state.p_pos})
        #
        # for i, agent in enumerate(self.agents):
        #     # Message = 'Users {} locats in {}'.format(i, landmark.state.p_pos)
        #     dict_state.append({'UAV {}'.format(i): agent.state.p_pos, 'band_asign': agent.state.c})
        # with open('./state_results/states.txt','a') as f:
        #     f.write(str(dict_state)+'\n')

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i, agent in enumerate(self.agents):
            if not agent.movable: continue
            agent.state.p_vel = agent.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                agent.state.p_vel += (p_force[i] / agent.mass) * self.dt
            if agent.max_speed is not None:
                speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
                if speed > agent.max_speed:
                    agent.state.p_vel = agent.state.p_vel / np.sqrt(np.square(agent.state.p_vel[0]) +
                                                                  np.square(agent.state.p_vel[1])) * agent.max_speed
            agent.state.p_pos += agent.state.p_vel * self.dt

    def update_landmark_pos(self):
        # for landmark in self.landmarks:
        #     if not landmark.movable: continue
        #     if landmark.state.hold_action > 0:
        #         landmark.state.hold_action -= 1
        #         landmark.state.p_vel = np.random.uniform(-1, +1, self.dim_p) * 0.5
        #     else:
        #         landmark.state.hold_action = np.random.randint(0,10)
        #
        #     temp_pos = landmark.state.p_pos + landmark.state.p_vel * self.dt
        #     while not (-1 <= temp_pos[0] <= 1 and -1 <= temp_pos[1] <= 1):
        #         landmark.state.p_vel = np.random.uniform(-1, +1, self.dim_p) * 0.5
        #         temp_pos = landmark.state.p_pos + landmark.state.p_vel * self.dt
        #         # print(temp_pos)
        #     landmark.state.p_pos = temp_pos

        for landmark in self.landmarks:
            if not landmark.movable: continue
            landmark.state.p_vel = np.random.uniform(-1, +1, self.dim_p) * 0.5
            temp_pos = landmark.state.p_pos + landmark.state.p_vel * self.dt
            while not (-1 <= temp_pos[0] <= 1 and -1 <= temp_pos[1] <= 1):
                landmark.state.p_vel = np.random.uniform(-1, +1, self.dim_p) * 0.5
                temp_pos = landmark.state.p_pos + landmark.state.p_vel * self.dt
                # print(temp_pos)
            landmark.state.p_pos = temp_pos


    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            # noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            # agent.state.c = agent.action.c + noise
            agent.state.c = agent.action.c

    def update_agent_pos(self):
        # set communication state (directly for now)
        for i, agent in enumerate(self.agents):
            if not agent.movable: continue
            noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
            agent.state.p_vel = (agent.action.u + noise) * (1 - self.damping)
            print(agent.state.p_vel)
            if agent.max_speed is not None:
                speed = np.sqrt(np.square(agent.state.p_vel[0]) + np.square(agent.state.p_vel[1]))
                if speed > agent.max_speed:
                    print('高于最大速度')
                    agent.state.p_vel = agent.state.p_vel / np.sqrt(np.square(agent.state.p_vel[0]) +
                                                                  np.square(agent.state.p_vel[1])) * agent.max_speed
            agent.state.p_pos += agent.state.p_vel * self.dt

    def update_world_state(self):
        # 根据基站提供的方案，用户进行选择

        bw = np.zeros((len(self.agents), self.dim_c))
        power = np.zeros((len(self.agents), self.dim_pow))
        disVec = np.zeros((len(self.agents), self.dim_c))
        height = np.ones((len(self.agents), self.dim_c)) * 200  # 无人机的飞行高度
        for index, agent in enumerate(self.agents):
            bw[index] = np.array(list(self.max_band * i for i in agent.action.c))
            power[index] = np.array(list(self.pow_UAVs * i for i in agent.action.pow))
            # 计算无人机到各个用户的欧式距离
            for i, landmark in enumerate(self.landmarks):
                disVec[index][i] = np.linalg.norm(agent.state.p_pos - landmark.state.p_pos, 2) * 1000
                # 距离超过定值时，无法进行通信，将带宽分配为0
                # bw[index][i] = 0.1 if disVec[index][i] > 0.025*8 else bw[index][i]
            # 接收到的信号功率
            signal_pow = power * self.gain_UAV / (disVec ** self.los_exp_UAV + height ** 2)

        # 接收端噪声功率
        noise_pow = bw * self.noise_pdf + 1e-9
        # 接收端信噪比
        SNR = signal_pow / noise_pow

        SNR[SNR < np.sqrt(10)] = 0 #SNR小于5dB时，无法通信，将其强制置零


        # for i in range(self.dim_c):
        #     for j in range(len(self.agents)):
        #         if 10*np.log10(SNR[j][i]) < 15:
        #             bw[j][i] = 0

        # 信息传输速率
        Rate = bw * np.log2(np.ones((len(self.agents), self.dim_c)) + SNR)

        ## 保证每个用户只与一个无人机建立通信链路
        mask = np.zeros((len(self.agents), self.dim_c)) #被遮掉的位置为0
        # mask = -1 * np.ones((len(self.agents), self.dim_c)) #被遮掉的位置为负数

        for i, landmark in enumerate(self.landmarks):
            if (landmark.assign_index is not None) and (landmark.assign_index != np.argmax(Rate[:, i])):
                landmark.switch_cout = 1
                landmark.assign_index = np.argmax(Rate[:, i])
            else:
                landmark.assign_index = np.argmax(Rate[:, i])

            mask[landmark.assign_index][i] = 1


        self.bandwidth = mask * bw
        self.tx_pow = mask * power
        self.Rate = mask * Rate
        self.SNR = 10*np.log10(mask * SNR + 1e-10)

        for i, agent in enumerate(self.agents):
            agent.state.assign = self.bandwidth[i, :]/self.max_band
            agent.state.UAV_Rate = self.Rate[i, :]
            agent.state.SNRforUser = SNR[i, :]
            agent.state.assign_pow = self.tx_pow[i, :]/self.pow_UAVs

        # for i, landmark in enumerate(self.landmarks):
        #     band = self.bandwidth.transpose()
        #     landmark.assign_index = np.nonzero(band[i])



    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # not a collider
        if (entity_a is entity_b):
            return [None, None] # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]