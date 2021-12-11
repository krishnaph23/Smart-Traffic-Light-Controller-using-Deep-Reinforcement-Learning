import traci
import numpy as np
import random
import timeit
import os

# phase codes based on environment.net.xml
PHASE_N_GREEN = 0
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4
PHASE_S_YELLOW = 5
PHASE_W_GREEN = 6
PHASE_W_YELLOW = 7


class Simulation:
    def __init__(self, Model1, Memory1, Model2, Memory2, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, training_epochs):
        self._Model1 = Model1
        self._Model2 = Model2

        self._Memory1 = Memory1
        self._Memory2 = Memory2

        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps  # 5400
        self._green_duration = green_duration  # 10
        self._yellow_duration = yellow_duration  # 4

        self._num_states = 40  # 80

        self._num_actions = 4  # 4
        
        self._reward_store1 = []
        self._reward_store2 = []

        self._reward_store = []

        self._cumulative_wait_store1 = []
        self._cumulative_wait_store2 = []

        self._cumulative_wait_store = []

        self._avg_queue_length_store1 = []
        self._avg_queue_length_store2 = []

        self._avg_queue_length_store = []

        self._training_epochs = training_epochs  # 800

    def run(self, episode, epsilon):
        start_time = timeit.default_timer()

        # first, generate the route file for this simulation and set up sumo
        self._TrafficGen.generate_routefile(seed=episode)
        traci.start(self._sumo_cmd)
        print("Simulating...")

        # inits
        self._step = 0
        self._waiting_times1 = {}
        self._waiting_times2 = {}

        self._sum_neg_reward1 = 0
        self._sum_neg_reward2 = 0
        self._sum_neg_reward = 0

        self._sum_queue_length1 = 0
        self._sum_queue_length2 = 0
        self._sum_queue_length = 0

        self._sum_waiting_time1 = 0
        self._sum_waiting_time2 = 0
        self._sum_waiting_time = 0

        old_total_wait1 = 0
        old_total_wait2 = 0

        old_state1 = -1
        old_action1 = -1
        old_state2 = -1
        old_action2 = -1

        while self._step < self._max_steps:
            current_state1, current_state2 = self._get_state()
            current_total_wait1, current_total_wait2 = self._collect_waiting_times()

            reward1 = old_total_wait1 - current_total_wait1
            reward2 = old_total_wait2 - current_total_wait2

            if self._step != 0:
                self._Memory1.add_sample((old_state1, old_action1, reward1, current_state1))
                self._Memory2.add_sample((old_state2, old_action2, reward2, current_state2))

            if random.random() < epsilon:
                action1 = random.randint(0, 3)
                action2 = random.randint(0, 3)
            else:
                action1 = np.argmax(self._Model1.predict_one(current_state1))
                action2 = np.argmax(self._Model2.predict_one(current_state2))

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action1 != action1:
                self._set_yellow_phase(old_action1, 'TL1')
            if self._step != 0 and old_action2 != action2:
                self._set_yellow_phase(old_action2, 'TL2')

            if self._step != 0:
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action1, 'TL1')
            self._set_green_phase(action2, 'TL2')

            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state1 = current_state1
            old_state2 = current_state2

            old_action1 = action1
            old_action2 = action2

            old_total_wait1 = current_total_wait1
            old_total_wait2 = current_total_wait2

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward1 < 0:
                self._sum_neg_reward1 += reward1
            if reward2 < 0:
                self._sum_neg_reward2 += reward2
            
        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward1 + self._sum_neg_reward2, "- Epsilon:", round(epsilon, 2))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("Training...")
        start_time = timeit.default_timer()
        for _ in range(self._training_epochs):
            self._replay()
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time

    def _simulate(self, steps_todo):
        """
        Execute steps in sumo while gathering statistics
        """
        if (self._step + steps_todo) >= self._max_steps:  # do not do more steps than the maximum allowed number of steps
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length1, queue_length2 = self._get_queue_length()

            self._sum_queue_length1 += queue_length1
            self._sum_queue_length2 += queue_length2

            self._sum_waiting_time1 += queue_length1
            self._sum_waiting_time2 += queue_length2

    def _collect_waiting_times(self):
        incoming_roads1 = ["A2TL1", "B2TL1", "TL22TL1", "F2TL1"]
        incoming_roads2 = ["C2TL2", "D2TL2", "E2TL2", "TL12TL2"]

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            # get the road id where the car is located
            road_id = traci.vehicle.getRoadID(car_id)

            if road_id in incoming_roads1:  # consider only the waiting times of cars in incoming roads
                self._waiting_times1[car_id] = wait_time
            else:
                if car_id in self._waiting_times1:  # a car that was tracked has cleared the intersection
                    del self._waiting_times1[car_id]

            if road_id in incoming_roads2:  # consider only the waiting times of cars in incoming roads
                self._waiting_times2[car_id] = wait_time
            else:
                if car_id in self._waiting_times2:  # a car that was tracked has cleared the intersection
                    del self._waiting_times2[car_id]

        total_waiting_time1 = sum(self._waiting_times1.values())
        total_waiting_time2 = sum(self._waiting_times2.values())

        return [total_waiting_time1, total_waiting_time2]

    def _set_yellow_phase(self, old_action, intersection):
        # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
        yellow_phase_code = old_action * 2 + 1
        traci.trafficlight.setPhase(intersection, yellow_phase_code)

    def _set_green_phase(self, action_number, intersection):
        """
        Activate the correct green light combination in sumo
        """
        if action_number == 0:
            traci.trafficlight.setPhase(intersection, PHASE_N_GREEN)
        elif action_number == 1:
            traci.trafficlight.setPhase(intersection, PHASE_E_GREEN)
        elif action_number == 2:
            traci.trafficlight.setPhase(intersection, PHASE_S_GREEN)
        else:
            traci.trafficlight.setPhase(intersection, PHASE_W_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N1 = traci.edge.getLastStepHaltingNumber("A2TL1")
        halt_S1 = traci.edge.getLastStepHaltingNumber("B2TL1")
        halt_E1 = traci.edge.getLastStepHaltingNumber("TL22TL1")
        halt_W1 = traci.edge.getLastStepHaltingNumber("F2TL1")

        halt_N2 = traci.edge.getLastStepHaltingNumber("C2TL2")
        halt_S2 = traci.edge.getLastStepHaltingNumber("D2TL2")
        halt_E2 = traci.edge.getLastStepHaltingNumber("E2TL2")
        halt_W2 = traci.edge.getLastStepHaltingNumber("TL12TL2")

        queue_length1 = halt_N1 + halt_S1 + halt_E1 + halt_W1
        queue_length2 = halt_N2 + halt_S2 + halt_E2 + halt_W2
        return [queue_length1, queue_length2]

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state1 = np.zeros(self._num_states)
        state2 = np.zeros(self._num_states)

        car_list = traci.vehicle.getIDList()

        for car_id in car_list:
            lane_pos = traci.vehicle.getLanePosition(car_id)
            lane_id = traci.vehicle.getLaneID(car_id)
            lane_pos = 500 - lane_pos

            # distance in meters from the traffic light -> mapping into cells
            if lane_pos < 7:
                lane_cell = 0
            elif lane_pos < 14:
                lane_cell = 1
            elif lane_pos < 21:
                lane_cell = 2
            elif lane_pos < 28:
                lane_cell = 3
            elif lane_pos < 40:
                lane_cell = 4
            elif lane_pos < 55:
                lane_cell = 5
            elif lane_pos < 80:
                lane_cell = 6
            elif lane_pos < 120:
                lane_cell = 7
            elif lane_pos < 240:
                lane_cell = 8
            elif lane_pos <= 500:
                lane_cell = 9

            if lane_id == "A2TL1_0" or lane_id == "A2TL1_1" or lane_id == "A2TL1_2" or lane_id == "A2TL1_3":
                lane_group = 10
            elif lane_id == "B2TL1_0" or lane_id == "B2TL1_1" or lane_id == "B2TL1_2" or lane_id == "B2TL1_3":
                lane_group = 11
            elif lane_id == "TL22TL1_0" or lane_id == "TL22TL1_1" or lane_id == "TL22TL1_2" or lane_id == "TL22TL1_3":
                lane_group = 12
            elif lane_id == "F2TL1_0" or lane_id == "F2TL1_1" or lane_id == "F2TL1_2" or lane_id == "F2TL1_3":
                lane_group = 13

            elif lane_id == "C2TL2_0" or lane_id == "C2TL2_1" or lane_id == "C2TL2_2" or lane_id == "C2TL2_3":
                lane_group = 20
            elif lane_id == "D2TL2_0" or lane_id == "D2TL2_1" or lane_id == "D2TL2_2" or lane_id == "D2TL2_3":
                lane_group = 21
            elif lane_id == "E2TL2_0" or lane_id == "E2TL2_1" or lane_id == "E2TL2_2" or lane_id == "E2TL2_3":
                lane_group = 22
            elif lane_id == "TL12TL2_0" or lane_id == "TL12TL2_1" or lane_id == "TL12TL2_2" or lane_id == "TL12TL2_3":
                lane_group = 23
            else:
                lane_group = -1

            if lane_group >= 10 and lane_group <= 13:
                lane_group -= 10
                car_position1 = int(str(lane_group) + str(lane_cell))
                valid_car1 = True
            else:
                valid_car1 = False

            if lane_group >= 20 and lane_group <= 23:
                lane_group -= 20
                car_position2 = int(str(lane_group) + str(lane_cell))
                valid_car2 = True
            else:
                valid_car2 = False

            if valid_car1:
                state1[car_position1] = 1
            elif valid_car2:
                state2[car_position2] = 1

        return [state1, state2]

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch1 = self._Memory1.get_samples(self._Model1.batch_size)
        batch2 = self._Memory2.get_samples(self._Model2.batch_size)

        if len(batch1) > 0:
            states = np.array([val[0] for val in batch1])
            next_states = np.array([val[3] for val in batch1])

            q_s_a = self._Model1.predict_batch(states)
            q_s_a_d = self._Model1.predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch1), self._num_states))
            y = np.zeros((len(batch1), self._num_actions))

            for i, b in enumerate(batch1):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * \
                    np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model1.train_batch(x, y)  # train the NN

        if len(batch2) > 0:
            states = np.array([val[0] for val in batch2])
            next_states = np.array([val[3] for val in batch2])

            q_s_a = self._Model2.predict_batch(states)
            q_s_a_d = self._Model2.predict_batch(next_states)

            x = np.zeros((len(batch2), self._num_states))
            y = np.zeros((len(batch2), self._num_actions))

            for i, b in enumerate(batch2):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * \
                    np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model2.train_batch(x, y)  # train the NN

    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store1.append(self._sum_neg_reward1)
        self._cumulative_wait_store1.append(self._sum_waiting_time1)
        self._avg_queue_length_store1.append(self._sum_queue_length1 / self._max_steps)

        self._reward_store2.append(self._sum_neg_reward2)
        self._cumulative_wait_store2.append(self._sum_waiting_time2)
        self._avg_queue_length_store2.append(self._sum_queue_length2 / self._max_steps)

        self._reward_store.append(self._sum_neg_reward1 + self._sum_neg_reward2)
        self._cumulative_wait_store.append(self._sum_waiting_time1 + self._sum_waiting_time2)
        self._avg_queue_length_store.append((self._sum_queue_length1 + self._sum_queue_length2) / self._max_steps)
