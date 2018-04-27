import gym
from gym import spaces
from gym.utils import seeding
from gym import utils
import numpy as np
import sys
from gym import Env

import os, sys


if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

# sys.path.append("/Users/cheryl/sumo-0.32.0/tools")

import traci


class FourIntersectsEnv(Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, traffic_light_junction_lanes_dict, version_num):
        self.seed()
        sumo_binary = "/Users/cheryl/sumo-0.32.0/bin/sumo-gui"
        sumo_config_file = 'sumo_env/four_intersects_' + str(version_num) + '.sumocfg'
        self.sumo_cmd = [sumo_binary, "-c", sumo_config_file, "-Q", "-S"]
        self.rl_algorithm = "RLTSC-1"
        # self.rl_algorithm = "RLTSC-2"
        self.sumo_step = 0
        self.sumo_running = False
        # self.state_vector = [0, 30, 0, 0]
        # self.state = self.convert_to_int(self.state_vector)
        self.traffic_light_junction_lanes_dict = traffic_light_junction_lanes_dict

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def send_action_to_env(self, junction_id_to_phase_pair):
        phase_switched = False
        # get junction_id and to_state_phase
        assert(len(junction_id_to_phase_pair) == 2)
        junction_id = junction_id_to_phase_pair[0]
        to_state_phase = junction_id_to_phase_pair[1] # 0 or 1

        previous_phase = traci.trafficlight.getPhase(str(junction_id))
        previous_phase_elapsed_time = traci.trafficlight.getNextSwitch(str(junction_id)) - \
                                      traci.simulation.getCurrentTime()
        previous_state_phase = previous_phase // 3

        is_switch = previous_state_phase != to_state_phase

        # to switch
        if is_switch:
            new_phase = (previous_phase + is_switch) % 6
            traci.trafficlight.setPhase(str(junction_id), new_phase)
            phase_switched = True
        # not switch
        else:
            # extend 1 second on the current phase, max is 30 seconds
            current_duration = traci.trafficlight.getPhaseDuration(str(junction_id))
            extended_duration = current_duration + 1
            if extended_duration > 30:
                extended_duration = 30
            traci.trafficlight.setPhaseDuration(str(junction_id), extended_duration + 1)
            phase_switched = False
        return phase_switched

    def get_state_and_queue_at_junction(self, junction_id):
        new_phase = traci.trafficlight.getPhase(str(junction_id))
        new_state_phase = new_phase//3
        # if in transition phase, set elapsed time to 30 sec
        # otherwise, set elapsed time properly
        if new_phase%3 != 0:
            new_phase_elapsed_time = 30000
        else:
            new_phase_elapsed_time = traci.trafficlight.getNextSwitch(str(junction_id)) - \
                                     traci.simulation.getCurrentTime()

        # update state vector
        self.state_vector[0]= new_state_phase
        self.state_vector[1] = new_phase_elapsed_time/1000

        max_queue_length = self.get_max_queue_length(junction_id)
        self.state_vector[2] = max_queue_length[0]
        self.state_vector[3] = max_queue_length[1]

        queue = self._queue_score(junction_id)

        state = self.convert_to_int(self.state_vector)

        # observation = self.observation()
        return state, queue

    def step(self, action):
        traci.simulationStep()
        self.sumo_step += 1
        observation = self.observation()
        done = self.sumo_step == 100000
        return observation, -1, done, {}

    def start_sumo(self):
        if not self.sumo_running:
            traci.start(self.sumo_cmd)
            self.sumo_step = 0
            self.sumo_running = True

    def stop_sumo(self):
        if self.sumo_running:
            traci.close()
            self.sumo_running = False

    def reset(self):
        self.stop_sumo()
        self.state_vector = [0, 30, 0, 0]
        self.state = self.convert_to_int(self.state_vector)
        self.start_sumo()

    def convert_to_int(self, state_vector):
        bases = [2, 31, 20, 20]
        assert(len(bases) == len(state_vector))
        res = 0;
        for i in range(len(state_vector)):
            res = res * bases[i] + state_vector[i]
        return int(res)

    def get_max_queue_length(self, junction_id):
        L_direction_queue = 0
        D_direction_queue = 0
        incoming_lanes = self.traffic_light_junction_lanes_dict[str(junction_id)]
        max_L_direction_queue = 0
        max_D_direction_queue = 0

        for lane in incoming_lanes:
            if lane.find('L') != -1 :
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = traci.lane.getLength(lane)
                for vehicle_id in vehicle_ids:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    if speed < 2.7:  # 2.7m/s approximately equals to 10km/h
                        L_direction_queue += 1
                L_direction_queue_length = L_direction_queue*5/lane_length
                if L_direction_queue_length > max_L_direction_queue:
                    max_L_direction_queue = L_direction_queue_length
            elif lane.find('D') != -1:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                lane_length = traci.lane.getLength(lane)
                for vehicle_id in vehicle_ids:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    if speed < 2.7:  # 2.7m/s approximately equals to 10km/h
                        D_direction_queue += 1
                D_direction_queue_length = D_direction_queue*5/lane_length
                if D_direction_queue_length > max_D_direction_queue:
                    max_D_direction_queue = D_direction_queue_length

        if max_L_direction_queue > 19:
            max_L_direction_queue = 19
        if max_D_direction_queue > 19:
            max_D_direction_queue = 19
        return [max_L_direction_queue, max_D_direction_queue]


    def _queue_score(self, junction_id):
        # average_queue = 0
        # waiting_time_all_junction = 0
        score = 0
        # if self.rl_algorithm == "RLTSC-1" or self.rl_algorithm == "RLTSC-3":
        #     # get average queue (travel below 10km/h) length
        #     # 1. getting all lanes inflow to the junction
        #     # 2. find cars that travel below 10km/h
        #     # 3. count the number of slow vechcles, then devide by the nunber of lanes
        #     slow_vehicle_count = 0
        #     incoming_lanes = self.traffic_light_junction_lanes_dict[str(junction_id)]
        #     for lane in incoming_lanes:
        #         vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
        #         for vehicle_id in vehicle_ids:
        #             speed = traci.vehicle.getSpeed(vehicle_id)
        #             if speed < 2.7: # 2.7m/s approximately equals to 10km/h
        #                 slow_vehicle_count += 1
        #     average_queue = slow_vehicle_count/len(incoming_lanes)
        # if self.rl_algorithm == "RLTSC-2" or self.rl_algorithm == "RLTSC-3":
        #     # sum of average waiting times (AWT) for each junction
        #     # 1. for each junction,
        #         # get all lanes inflow to the junction
        #         # waiting time of all lanes. divide by the number of lanes
        #     # 2. sum all these up
        #     waiting_time_all_junction = 0
        #     for junction_id, incoming_lanes in self.traffic_light_junction_lanes_dict.items():
        #         waiting_time_at_junction_sum = 0
        #         for lane in incoming_lanes:
        #             waiting_time_at_junction_sum += traci.lane.getWaitingTime(lane)
        #         waiting_time_junction_average = waiting_time_at_junction_sum/len(incoming_lanes)
        #         waiting_time_all_junction += waiting_time_junction_average
        #
        # score = average_queue + waiting_time_all_junction
        if self.rl_algorithm == "RLTSC-1":
            max_queue_length = self.get_max_queue_length(junction_id)
            score  = max_queue_length[0]**2 + max_queue_length[1]**2
            return score
        elif self.rl_algorithm == "RLTSC-2":
            queue_sum = 0
            for junction_id, incoming_lanes in self.traffic_light_junction_lanes_dict.items():
                max_queue_length = self.get_max_queue_length(junction_id)
                score = max_queue_length[0] ** 2 + max_queue_length[1] ** 2
                queue_sum += score
            return queue_sum

    def observation(self):
        queue_all_junction = 0
        for junction_id, incoming_lanes in self.traffic_light_junction_lanes_dict.items():
            slow_vehicle_count = 0
            for lane in incoming_lanes:
                vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
                for vehicle_id in vehicle_ids:
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    if speed < 2.7:  # 2.7m/s approximately equals to 10km/h
                        slow_vehicle_count += 1
            queue_junction_average = slow_vehicle_count / len(incoming_lanes)
            queue_all_junction += queue_junction_average

        waiting_time_all_junction = 0
        for junction_id, incoming_lanes in self.traffic_light_junction_lanes_dict.items():
            waiting_time_at_junction_sum = 0
            for lane in incoming_lanes:
                waiting_time_at_junction_sum += traci.lane.getWaitingTime(lane)
            waiting_time_junction_average = waiting_time_at_junction_sum / len(incoming_lanes)
            waiting_time_all_junction += waiting_time_junction_average

        return queue_all_junction, waiting_time_all_junction

