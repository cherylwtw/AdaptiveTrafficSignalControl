import gym
from gym import spaces
from gym.utils import seeding
from gym import utils
import numpy as np
import sys
from gym import Env
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def get_traffic_light_phase_dict():
    phase_dict = {}
    source_net_file = '/Users/cheryl/Documents/Python/767FinalProject/four_intersects/sumo_env/four_intersects.net.xml'
    tree = ET.parse(source_net_file)
    root = tree.getroot()
    for junction in root.findall('tlLogic'):
        junction_id = junction.get('id')
        duration_list = []
        for phase in junction.findall('phase'):
            duration = phase.get('duration')
            duration_list.append(duration)
            phase_dict[junction_id] = duration_list
    return phase_dict

def convert_to_int(state_vector):
    bases = [2, 31, 20, 20]
    assert(len(bases) == len(state_vector))
    res = 0;
    for i in range(len(state_vector)):
        res = res * bases[i] + state_vector[i]
    return int(res)

class Q_Learning_Agent:
    def __init__(self, junction_id, nb_states, nb_actions, n, gamma=0.9):
        # self.game_env = game_env
        self.junction_id = junction_id
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        # initial/current state
        self.gamma = gamma
        self.state_vector = [0, 30, 0, 0]
        self.state = convert_to_int(self.state_vector)
        self.previous_queue = 0
        self.current_queue = 0
        self.n = n
        self.count_down_to_next_decision = 0
        self.action = 0
        # initialize something, found out that using random is much better than zeros
        self.state_action = np.zeros((nb_states, nb_actions))

    def breaktie_argmax(self, vector):
        return np.random.choice(np.where(vector == vector.max())[0])

    def make_decision(self, game_env, epsilon): # to_phase = 0 or 1 in our environment
        if np.random.rand() < 1-epsilon:
            self.action = np.random.randint(self.nb_actions)
        else:
            self.action = self.breaktie_argmax(self.state_action[self.state, :])

        state, queue = game_env.env.get_state_and_queue_at_junction(self.junction_id)
        self.previous_queue = queue
        phase_switched = game_env.env.send_action_to_env([self.junction_id, self.action])
        if phase_switched is True:
            self.count_down_to_next_decision = 25

    def learning(self, game_env, alpha=0.05):
        if self.count_down_to_next_decision == 0:
            next_state, queue = game_env.env.get_state_and_queue_at_junction(self.junction_id)

        self.current_queue = queue
        reward = self.previous_queue - self.current_queue

        # direct rl
        self.state_action[self.state, self.action] += alpha * (
            reward + self.gamma * np.max(self.state_action[next_state, :]) - self.state_action[
                self.state, self.action])
        self.state = next_state

    def step(self, game_env):
        self.action = self.breaktie_argmax(self.state_action[self.state, :])
        phase_switched = game_env.env.send_action_to_env([self.junction_id, self.action])
        if phase_switched is True:
            self.count_down_to_next_decision = 25


class Dyna_Agent:
    def __init__(self, junction_id, nb_states, nb_actions, n, gamma=0.9):
        # self.game_env = game_env
        self.junction_id = junction_id
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        # initial/current state
        self.gamma = gamma
        self.state_vector = [0, 30, 0, 0]
        self.state = convert_to_int(self.state_vector)
        self.previous_queue = 0
        self.current_queue = 0
        self.n = n
        self.count_down_to_next_decision = 0
        self.action = 0
        # initialize something, found out that using random is much better than zeros
        self.state_action = np.random.rand(nb_states, nb_actions)
        # assuming a deterministic environment, therefore only destination is stored rather than a distribution of state space
        self.model_trans = np.zeros((nb_states, nb_actions), dtype=int)
        self.model_reward = np.zeros((nb_states, nb_actions))

    def breaktie_argmax(self, vector):
        return np.random.choice(np.where(vector == vector.max())[0])

    def make_decision(self, game_env, epsilon): # to_phase = 0 or 1 in our environment
        if np.random.rand() < 1-epsilon:
            self.action = np.random.randint(self.nb_actions)
        else:
            self.action = self.breaktie_argmax(self.state_action[self.state, :])

        state, queue = game_env.env.get_state_and_queue_at_junction(self.junction_id)
        self.previous_queue = queue
        phase_switched = game_env.env.send_action_to_env([self.junction_id, self.action])
        if phase_switched is True:
            self.count_down_to_next_decision = 25



    def learning(self, game_env, alpha=0.05):
        if self.count_down_to_next_decision == 0:
            next_state, queue = game_env.env.get_state_and_queue_at_junction(self.junction_id)

        self.current_queue = queue
        reward = self.previous_queue - self.current_queue

        # direct rl
        self.state_action[self.state, self.action] += alpha * (
            reward + self.gamma * np.max(self.state_action[next_state, :]) - self.state_action[
                self.state, self.action])
        # update knowledge of the model
        self.model_trans[self.state, self.action] = next_state
        self.model_reward[self.state, self.action] = reward
        self.state = next_state
        # dyna-q planning, the following codes and logic rely heavily on the deterministic environment guarantees;
        # also under the assumption that state 0 is not reachable.
        for i in range(self.n):
            fantasied_state = np.random.choice(np.where(self.model_trans.sum(axis=1) != 0)[0])
            fantasied_action = np.random.choice(np.where(self.model_trans[fantasied_state, :] != 0)[0])
            fantasied_next_state = self.model_trans[fantasied_state, fantasied_action]
            fantansied_reward = self.model_reward[fantasied_state, fantasied_action]
            self.state_action[fantasied_state, fantasied_action] += alpha * (
                fantansied_reward + self.gamma * np.max(self.state_action[fantasied_next_state, :]) -
                self.state_action[
                    fantasied_state, fantasied_action])

    def step(self, game_env):
        self.action = self.breaktie_argmax(self.state_action[self.state, :])
        phase_switched = game_env.env.send_action_to_env([self.junction_id, self.action])
        if phase_switched is True:
            self.count_down_to_next_decision = 25

def fixed_control_run(iterations):
    queue_length_record_multi = []
    waiting_time_record_multi = []

    for i in range(iterations):
        queue_length_record = []
        waiting_time_record = []

        env_name = 'FourIntersects-v' + str(i + 1)
        game_env = gym.make(env_name)
        game_env.reset()

        time_step = 0
        while time_step < 10000:
            # print('time step is ' + str(time_step))
            observation, reward, done, _ = game_env.step(-1)
            queue_length_record.append(observation[0])
            waiting_time_record.append(observation[1])
            time_step += 1

        np_queue_length_record = np.asarray(queue_length_record)
        np_waiting_time_record = np.asarray(waiting_time_record)
        queue_length_record_multi.append(np_queue_length_record)
        waiting_time_record_multi.append(np_waiting_time_record)

    queue_length_record_multi = np.array(queue_length_record_multi)
    waiting_time_record_multi = np.array(waiting_time_record_multi)

    queue_length_average = np.mean(queue_length_record_multi.mean(axis=0)[-10000:])
    waiting_time_average = np.mean(waiting_time_record_multi.mean(axis=0)[-10000:])
    print('fixed control')
    print('after ' + str(iterations) + ' iterations:')
    print('queue_length_average = ' + str(queue_length_average))
    print('waiting_time_average = ' + str(waiting_time_average))

    # # graph
    # plt.figure()
    # plt.plot(queue_length_record_multi.mean(axis=0), label='Fixed Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Queue Length')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(waiting_time_record_multi.mean(axis=0), label='Fixed Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Waiting Time')
    # plt.legend()
    #
    # plt.show()

def q_learning_control_run(iterations):
    queue_length_record_multi = []
    waiting_time_record_multi = []

    for i in range(iterations):
        queue_length_record = []
        waiting_time_record = []

        # first run is for learning
        env_name = 'FourIntersects-v' + str(i + 1)
        game_env = gym.make(env_name)
        game_env.reset()
        state_num = 2*31*20*20

        # create agent dict
        agent_dict = {}
        traffic_light_phase_dict = get_traffic_light_phase_dict()

        for junction_id, duration_list in traffic_light_phase_dict.items():
            phase_num = 2
            state_num = phase_num * 31 * 20**phase_num
            agent = Q_Learning_Agent(junction_id, state_num, 2, 50)
            agent_dict[junction_id] = agent

        # the first 50000 steps are training
        epsilon = 0.9
        alpha = 0.05
        for time_step in range(10000):
            for junction_id, agent in agent_dict.items():
                if agent.count_down_to_next_decision == 0:  # otherwise the light has been recently set
                    agent.learning(game_env, alpha=alpha)
                    agent.make_decision(game_env, epsilon=epsilon)
                else:
                    agent.count_down_to_next_decision -= 1
            obseration, reward, done, _ = game_env.step(-1)

            queue_length_record.append(obseration[0])
            waiting_time_record.append(obseration[1])

        # print('done learning')

        # if i == 0:
        #     for junction_id, agent in agent_dict.items():
        #         agent.state = game_env.env.state

        # random flow starts here
        # for time_step in range(10000):
        #     for junction_id, agent in agent_dict.items():
        #         if agent.count_down_to_next_decision == 0:  # otherwise the light has been recently set
        #             agent.step(game_env)
        #         else:
        #             agent.count_down_to_next_decision -= 1
        #     # at each time step
        #     observation, reward, done, _ = game_env.step(-1)
        #
        #     # overwrite agent.state, so next action is taken based on the current state
        #     for junction_id, agent in agent_dict.items():
        #         agent.state, _ = game_env.env.get_state_and_queue_at_junction(junction_id)
        #
        #     queue_length_record.append(observation[0])
        #     waiting_time_record.append(observation[1])

        np_queue_length_record = np.asarray(queue_length_record)
        np_waiting_time_record = np.asarray(waiting_time_record)
        queue_length_record_multi.append(np_queue_length_record)
        waiting_time_record_multi.append(np_waiting_time_record)

    queue_length_record_multi = np.array(queue_length_record_multi)
    waiting_time_record_multi = np.array(waiting_time_record_multi)

    queue_length_average = np.mean(queue_length_record_multi.mean(axis=0)[-10000:])
    waiting_time_average = np.mean(waiting_time_record_multi.mean(axis=0)[-10000:])
    print('q learning')
    print('after ' + str(iterations) + ' iterations:')
    print('queue_length_average = ' + str(queue_length_average))
    print('waiting_time_average = ' + str(waiting_time_average))


    # graph
    # plt.figure()
    # plt.plot(queue_length_record_multi.mean(axis=0), label='Q Learning Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Queue Length')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(waiting_time_record_multi.mean(axis=0), label='Q Learning Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Waiting Time')
    # plt.legend()
    #
    # plt.show()


def dyna_control_run(iterations):
    queue_length_record_multi = []
    waiting_time_record_multi = []

    for i in range(iterations):
        queue_length_record = []
        waiting_time_record = []

        # first run is for learning
        env_name = 'FourIntersects-v' + str(i + 1)
        game_env = gym.make(env_name)
        game_env.reset()
        state_num = 2*31*20*20

        # create agent dict
        agent_dict = {}
        traffic_light_phase_dict = get_traffic_light_phase_dict()

        for junction_id, duration_list in traffic_light_phase_dict.items():
            phase_num = 2
            state_num = phase_num * 31 * 20**phase_num
            agent = Dyna_Agent(junction_id, state_num, 2, 50)
            agent_dict[junction_id] = agent

        # the first 50000 steps are training
        epsilon = 0.9
        alpha = 0.05
        for time_step in range(10000):
            for junction_id, agent in agent_dict.items():
                if agent.count_down_to_next_decision == 0:  # otherwise the light has been recently set
                    agent.learning(game_env, alpha=alpha)
                    agent.make_decision(game_env, epsilon=epsilon)
                else:
                    agent.count_down_to_next_decision -= 1
            obseration, reward, done, _ = game_env.step(-1)

            queue_length_record.append(obseration[0])
            waiting_time_record.append(obseration[1])

        # print('done learning')

        # for junction_id, agent in agent_dict.items():
        #     agent.state = game_env.env.state

        # random flow starts here
        # for time_step in range(3000):
        #     for junction_id, agent in agent_dict.items():
        #         if agent.count_down_to_next_decision == 0:  # otherwise the light has been recently set
        #             agent.step(game_env)
        #         else:
        #             agent.count_down_to_next_decision -= 1
        #     # at each time step
        #     observation, reward, done, _ = game_env.step(-1)
        #
        #     # overwrite agent.state, so next action is taken based on the current state
        #     for junction_id, agent in agent_dict.items():
        #         agent.state, _ = game_env.env.get_state_and_queue_at_junction(junction_id)
        #
        #     queue_length_record.append(observation[0])
        #     waiting_time_record.append(observation[1])

        np_queue_length_record = np.asarray(queue_length_record)
        np_waiting_time_record = np.asarray(waiting_time_record)
        queue_length_record_multi.append(np_queue_length_record)
        waiting_time_record_multi.append(np_waiting_time_record)

    queue_length_record_multi = np.array(queue_length_record_multi)
    waiting_time_record_multi = np.array(waiting_time_record_multi)

    queue_length_average = np.mean(queue_length_record_multi.mean(axis=0)[-10000:])
    waiting_time_average = np.mean(waiting_time_record_multi.mean(axis=0)[-10000:])
    print('Dyna Q')
    print('after ' + str(iterations) + ' iterations:')
    print('queue_length_average = ' + str(queue_length_average))
    print('waiting_time_average = ' + str(waiting_time_average))


    # # graph
    # plt.figure()
    # plt.plot(queue_length_record_multi.mean(axis=0), label='Dyna Learning Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Queue Length')
    # plt.legend()
    #
    # plt.figure()
    # plt.plot(waiting_time_record_multi.mean(axis=0), label='Dyna Learning Control')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Waiting Time')
    # plt.legend()

    # plt.show()

def main():
    fixed_control_run(10)
    q_learning_control_run(10)
    dyna_control_run(10)




if __name__ == "__main__":
    main()
