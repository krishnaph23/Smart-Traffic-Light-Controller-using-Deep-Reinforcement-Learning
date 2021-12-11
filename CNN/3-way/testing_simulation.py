import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_NS_GREEN = 0  # action 0 code 00
PHASE_NS_YELLOW = 1
PHASE_NSL_GREEN = 2  # action 1 code 01
PHASE_NSL_YELLOW = 3
PHASE_EW_GREEN = 4  # action 2 code 10
PHASE_EW_YELLOW = 5


class Simulation:
    def __init__(self, Model, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration, num_states, num_actions):
        self._Model = Model
        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states
        self._num_actions = num_actions
        self._reward_episode = []
        self._queue_length_episode = []

    def run(self, episode):
        """
        Runs the testing simulation
        """
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times = {}
        old_total_wait = 0
        old_action = -1 # dummy init

        while self._step < self._max_steps:
            
            # get current state of the intersection
            current_state = self._get_state()

            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state)

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_action = action
            old_total_wait = current_total_wait

            self._reward_episode.append(reward)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time


    def _simulate(self, steps_todo):
        """
        Proceed with the simulation in sumo
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length = self._get_queue_length() 
            self._queue_length_episode.append(queue_length)


    def _collect_waiting_times(self):
        """
        Retrieve the waiting time of every car in the incoming roads
        """
        incoming_roads = ["A2TL", "B2TL", "C2TL"]
        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located
            if road_id in incoming_roads:  # consider only the waiting times of cars in incoming roads
                self._waiting_times[car_id] = wait_time
            else:
                if car_id in self._waiting_times: # a car that was tracked has cleared the intersection
                    del self._waiting_times[car_id]
        total_waiting_time = sum(self._waiting_times.values())
        return total_waiting_time


    def _choose_action(self, state):
        """
        Pick the best action known based on the current state of the env
        """
        return np.argmax(self._Model.predict_one(state))


    def _set_yellow_phase(self, old_action):
        """
        Activate the correct yellow light combination in sumo
        """
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        traci.trafficlight.setPhase("TL", yellow_phase_code)


    def _set_green_phase(self, action_number):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase("TL", PHASE_NS_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase("TL", PHASE_NSL_GREEN)
        else:
            traci.trafficlight.setPhase("TL", PHASE_EW_GREEN)


    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N = traci.edge.getLastStepHaltingNumber("A2TL")
        halt_S = traci.edge.getLastStepHaltingNumber("B2TL")
        halt_E = traci.edge.getLastStepHaltingNumber("C2TL")
        queue_length = halt_N + halt_S + halt_E
        return queue_length



    def _get_state(self):
            positionMatrix = []
            velocityMatrix = []

            cellLength = 7
            offset = 11
            speedLimit = 14

            junctionPosition = traci.junction.getPosition('TL')[0]
            vehicles_road1 = traci.edge.getLastStepVehicleIDs('A2TL')
            vehicles_road2 = traci.edge.getLastStepVehicleIDs('B2TL')
            vehicles_road3 = traci.edge.getLastStepVehicleIDs('C2TL')
            for i in range(12):
                positionMatrix.append([])
                velocityMatrix.append([])
                for j in range(10):
                    positionMatrix[i].append(0)
                    velocityMatrix[i].append(0)

            for v in vehicles_road1:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[3 - traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

            for v in vehicles_road2:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[4 + traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                    velocityMatrix[4 + traci.vehicle.getLaneIndex(
                        v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            junctionPosition = traci.junction.getPosition('TL')[1]
            for v in vehicles_road3:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[8 + 3 -
                                traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[8 + 3 - traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit


            position = np.array(positionMatrix)
            position = position.reshape(1, 12, 10, 1)

            velocity = np.array(velocityMatrix)
            velocity = velocity.reshape(1, 12, 10, 1)

            return [position, velocity]


    @property
    def queue_length_episode(self):
        return self._queue_length_episode


    @property
    def reward_episode(self):
        return self._reward_episode
