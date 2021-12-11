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
PHASE_J_GREEN = 8
PHASE_J_YELLOW = 9

class Simulation:
    def __init__(self, Model1, Memory1, Model2, Memory2, Model3, Memory3, Model4, Memory4, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, training_epochs):
        self._Model1 = Model1
        self._Model2 = Model2
        self._Model3 = Model3
        self._Model4 = Model4

        self._Memory1 = Memory1
        self._Memory2 = Memory2
        self._Memory3 = Memory3
        self._Memory4 = Memory4

        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps  # 5400
        self._green_duration = green_duration #10
        self._yellow_duration = yellow_duration #4

        self._num_states1 = 40   #80
        self._num_states23 = 30   #80
        self._num_states4 = 50   #80

        self._num_actions1 = 4 #4
        self._num_actions23 = 3 #4
        self._num_actions4 = 5 #4

        self._reward_store1 = []
        self._reward_store2 = []
        self._reward_store3 = []
        self._reward_store4 = []
        self._reward_store = []

        self._cumulative_wait_store1 = []
        self._cumulative_wait_store2 = []
        self._cumulative_wait_store3 = []
        self._cumulative_wait_store4 = []
        self._cumulative_wait_store = []

        self._avg_queue_length_store1 = []
        self._avg_queue_length_store2 = []
        self._avg_queue_length_store3 = []
        self._avg_queue_length_store4 = []
        self._avg_queue_length_store = []

        self._training_epochs = training_epochs #800

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
        self._waiting_times3 = {}
        self._waiting_times4 = {}

        self._sum_neg_reward1 = 0
        self._sum_neg_reward2 = 0
        self._sum_neg_reward3 = 0
        self._sum_neg_reward4 = 0

        self._sum_queue_length1 = 0
        self._sum_queue_length2 = 0
        self._sum_queue_length3 = 0
        self._sum_queue_length4 = 0

        self._sum_waiting_time1 = 0
        self._sum_waiting_time2 = 0
        self._sum_waiting_time3 = 0
        self._sum_waiting_time4 = 0

        old_total_wait1 = 0
        old_total_wait2 = 0
        old_total_wait3 = 0
        old_total_wait4 = 0

        old_state1 = -1
        old_action1 = -1
        old_state2 = -1
        old_action2 = -1
        old_state3 = -1
        old_action3 = -1
        old_state4 = -1
        old_action4 = -1

        while self._step < self._max_steps:
            current_state1, current_state2, current_state3, current_state4 = self._get_state()
            current_total_wait1, current_total_wait2, current_total_wait3, current_total_wait4 = self._collect_waiting_times()

            reward1 = old_total_wait1 - current_total_wait1
            reward2 = old_total_wait2 - current_total_wait2
            reward3 = old_total_wait3 - current_total_wait3
            reward4 = old_total_wait4 - current_total_wait4

            if self._step != 0:
                self._Memory1.add_sample((old_state1, old_action1, reward1, current_state1))
                self._Memory2.add_sample((old_state2, old_action2, reward2, current_state2))
                self._Memory3.add_sample((old_state3, old_action3, reward3, current_state3))
                self._Memory4.add_sample((old_state4, old_action4, reward4, current_state4))

            if random.random() < epsilon:
                action1 = random.randint(0, 3)
                action2 = random.randint(0, 2)
                action3 = random.randint(0, 2)
                action4 = random.randint(0, 4)
            else:
                action1 = np.argmax(self._Model1.predict_one(current_state1))
                action2 = np.argmax(self._Model2.predict_one(current_state2))
                action3 = np.argmax(self._Model3.predict_one(current_state3))
                action4 = np.argmax(self._Model4.predict_one(current_state4))

            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action1 != action1:
                self._set_yellow_phase(old_action1, 'TL1')
            if self._step != 0 and old_action2 != action2:
                self._set_yellow_phase(old_action2, 'TL2')
            if self._step != 0 and old_action3 != action3:
                self._set_yellow_phase(old_action3, 'TL3')
            if self._step != 0 and old_action4 != action4:
                self._set_yellow_phase(old_action4, 'TL4')

            if self._step != 0:
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action1, 'TL1')
            self._set_green_phase(action2, 'TL2')
            self._set_green_phase(action3, 'TL3')
            self._set_green_phase(action4, 'TL4')

            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state1 = current_state1
            old_state2 = current_state2
            old_state3 = current_state3
            old_state4 = current_state4

            old_action1 = action1
            old_action2 = action2
            old_action3 = action3
            old_action4 = action4

            old_total_wait1 = current_total_wait1
            old_total_wait2 = current_total_wait2
            old_total_wait3 = current_total_wait3
            old_total_wait4 = current_total_wait4

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward1 < 0:
                self._sum_neg_reward1 += reward1
            if reward2 < 0:
                self._sum_neg_reward2 += reward2
            if reward3 < 0:
                self._sum_neg_reward3 += reward3
            if reward4 < 0:
                self._sum_neg_reward4 += reward4

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward1 + self._sum_neg_reward2 + self._sum_neg_reward3 + self._sum_neg_reward4, "- Epsilon:", round(epsilon, 2))
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
            self._step += 1 # update the step counter
            steps_todo -= 1
            queue_length1, queue_length2, queue_length3, queue_length4 = self._get_queue_length()

            self._sum_queue_length1 += queue_length1
            self._sum_queue_length2 += queue_length2
            self._sum_queue_length3 += queue_length3
            self._sum_queue_length4 += queue_length4

            self._sum_waiting_time1 += queue_length1 # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_waiting_time2 += queue_length2 # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_waiting_time3 += queue_length3 # 1 step while waiting in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds
            self._sum_waiting_time4 += queue_length4

    def _collect_waiting_times(self):
        incoming_roads1 = ["A2TL1", "B2TL1", "C2TL1", "TL22TL1"]
        incoming_roads2 = ["TL12TL2", "TL32TL2", "TL42TL2"]
        incoming_roads3 = ["G2TL3", "TL22TL3", "TL42TL3"]
        incoming_roads4 = ["D2TL4", "E2TL4", "F2TL4", "TL22TL4", "TL32TL4"]

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
            road_id = traci.vehicle.getRoadID(car_id)  # get the road id where the car is located

            if road_id in incoming_roads1:  # consider only the waiting times of cars in incoming roads
                self._waiting_times1[car_id] = wait_time
            else:
                if car_id in self._waiting_times1: # a car that was tracked has cleared the intersection
                    del self._waiting_times1[car_id]

            if road_id in incoming_roads2:  # consider only the waiting times of cars in incoming roads
                self._waiting_times2[car_id] = wait_time
            else:
                if car_id in self._waiting_times2:  # a car that was tracked has cleared the intersection
                    del self._waiting_times2[car_id]

            if road_id in incoming_roads3:  # consider only the waiting times of cars in incoming roads
                self._waiting_times3[car_id] = wait_time
            else:
                if car_id in self._waiting_times3:  # a car that was tracked has cleared the intersection
                    del self._waiting_times3[car_id]

            if road_id in incoming_roads4:  # consider only the waiting times of cars in incoming roads
                self._waiting_times4[car_id] = wait_time
            else:
                if car_id in self._waiting_times4:  # a car that was tracked has cleared the intersection
                    del self._waiting_times4[car_id]

        total_waiting_time1 = sum(self._waiting_times1.values())
        total_waiting_time2 = sum(self._waiting_times2.values())
        total_waiting_time3 = sum(self._waiting_times3.values())
        total_waiting_time4 = sum(self._waiting_times4.values())

        return [total_waiting_time1, total_waiting_time2, total_waiting_time3, total_waiting_time4]


    def _set_yellow_phase(self, old_action, intersection):
        yellow_phase_code = old_action * 2 + 1 # obtain the yellow phase code, based on the old action (ref on environment.net.xml)
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
        elif action_number == 3:
            traci.trafficlight.setPhase(intersection, PHASE_W_GREEN)
        else:
            traci.trafficlight.setPhase(intersection, PHASE_J_GREEN)

    def _get_queue_length(self):
        """
        Retrieve the number of cars with speed = 0 in every incoming lane
        """
        halt_N1 = traci.edge.getLastStepHaltingNumber("A2TL1")
        halt_S1 = traci.edge.getLastStepHaltingNumber("B2TL1")
        halt_E1 = traci.edge.getLastStepHaltingNumber("TL22TL1")
        halt_W1 = traci.edge.getLastStepHaltingNumber("C2TL1")

        halt_N2 = traci.edge.getLastStepHaltingNumber("TL12TL2")
        halt_S2 = traci.edge.getLastStepHaltingNumber("TL32TL2")
        halt_E2 = traci.edge.getLastStepHaltingNumber("TL42TL2")

        halt_N3 = traci.edge.getLastStepHaltingNumber("G2TL3")
        halt_S3 = traci.edge.getLastStepHaltingNumber("TL22TL3")
        halt_E3 = traci.edge.getLastStepHaltingNumber("TL42TL3")

        halt_N4 = traci.edge.getLastStepHaltingNumber("TL22TL4")
        halt_S4 = traci.edge.getLastStepHaltingNumber("D2TL4")
        halt_E4 = traci.edge.getLastStepHaltingNumber("E2TL4")
        halt_W4 = traci.edge.getLastStepHaltingNumber("F2TL4")
        halt_J4 = traci.edge.getLastStepHaltingNumber("TL32TL4")

        queue_length1 = halt_N1 + halt_S1 + halt_E1 + halt_W1
        queue_length2 = halt_N2 + halt_S2 + halt_E2
        queue_length3 = halt_N3 + halt_S3 + halt_E3
        queue_length4 = halt_N4 + halt_S4 + halt_E4 + halt_W4 + halt_J4

        return [queue_length1, queue_length2, queue_length3, queue_length4]

    def _get_state(self):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        state1 = np.zeros(self._num_states1)
        state2 = np.zeros(self._num_states23)
        state3 = np.zeros(self._num_states23)
        state4 = np.zeros(self._num_states4)

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
            elif lane_id == "C2TL1_0" or lane_id == "C2TL1_1" or lane_id == "C2TL1_2" or lane_id == "C2TL1_3":
                lane_group = 11
            elif lane_id == "TL22TL1_0" or lane_id == "TL22TL1_1" or lane_id == "TL22TL1_2" or lane_id == "TL22TL1_3":
                lane_group = 12
            elif lane_id == "B2TL1_0" or lane_id == "B2TL1_1" or lane_id == "B2TL1_2" or lane_id == "B2TL1_3":
                lane_group = 13

            elif lane_id == "TL12TL2_0" or lane_id == "TL12TL2_1" or lane_id == "TL12TL2_2" or lane_id == "TL12TL2_3":
                lane_group = 20
            elif lane_id == "TL32TL2_0" or lane_id == "TL32TL2_1" or lane_id == "TL32TL2_2" or lane_id == "TL32TL2_3":
                lane_group = 21
            elif lane_id == "TL42TL2_0" or lane_id == "TL42TL2_1" or lane_id == "TL42TL2_2" or lane_id == "TL42TL2_3":
                lane_group = 22

            elif lane_id == "G2TL3_0" or lane_id == "G2TL3_1" or lane_id == "G2TL3_2" or lane_id == "G2TL3_3":
                lane_group = 30
            elif lane_id == "TL22TL3_0" or lane_id == "TL22TL3_1" or lane_id == "TL22TL3_2" or lane_id == "TL22TL3_3":
                lane_group = 31
            elif lane_id == "TL42TL3_0" or lane_id == "TL42TL3_1" or lane_id == "TL42TL3_2" or lane_id == "TL42TL3_3":
                lane_group = 32

            elif lane_id == "D2TL4_0" or lane_id == "D2TL4_1" or lane_id == "D2TL4_2" or lane_id == "D2TL4_3":
                lane_group = 40
            elif lane_id == "E2TL4_0" or lane_id == "E2TL4_1" or lane_id == "E2TL4_2" or lane_id == "E2TL4_3":
                lane_group = 41
            elif lane_id == "F2TL4_0" or lane_id == "F2TL4_1" or lane_id == "F2TL4_2" or lane_id == "F2TL4_3":
                lane_group = 42
            elif lane_id == "TL32TL4_0" or lane_id == "TL32TL4_1" or lane_id == "TL32TL4_2" or lane_id == "TL32TL4_3":
                lane_group = 43
            elif lane_id == "TL22TL4_0" or lane_id == "TL22TL4_1" or lane_id == "TL22TL4_2" or lane_id == "TL22TL4_3":
                lane_group = 44
            else:
                lane_group = -1

            if lane_group >= 10 and lane_group <= 13:
                lane_group -= 10
                car_position1 = int(str(lane_group) + str(lane_cell))
                valid_car1 = True
            else:
                valid_car1 = False

            if lane_group >= 20 and lane_group <= 22:
                lane_group -= 20
                car_position2 = int(str(lane_group) + str(lane_cell))
                valid_car2 = True
            else:
                valid_car2 = False

            if lane_group >= 30 and lane_group <= 32:
                lane_group -= 30
                car_position3 = int(str(lane_group) + str(lane_cell))
                valid_car3 = True
            else:
                valid_car3 = False

            if lane_group >= 40 and lane_group <= 44:
                lane_group -= 40
                car_position4 = int(str(lane_group) + str(lane_cell))
                valid_car4 = True
            else:
                valid_car4 = False

            if valid_car1:
                state1[car_position1] = 1
            elif valid_car2:
                state2[car_position2] = 1
            elif valid_car3:
                state3[car_position3] = 1
            elif valid_car4:
                state4[car_position4] = 1

        return [state1, state2, state3, state4]

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch1 = self._Memory1.get_samples(self._Model1.batch_size)
        batch2 = self._Memory2.get_samples(self._Model2.batch_size)
        batch3 = self._Memory3.get_samples(self._Model3.batch_size)
        batch4 = self._Memory4.get_samples(self._Model4.batch_size)

        if len(batch1) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch1])  # extract states from the batch
            next_states = np.array([val[3] for val in batch1])  # extract next states from the batch

            # prediction
            q_s_a = self._Model1.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model1.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch1), self._num_states1))
            y = np.zeros((len(batch1), self._num_actions1))

            for i, b in enumerate(batch1):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model1.train_batch(x, y)  # train the NN

        if len(batch2) > 0:  # if the memory is full enough
            # extract states from the batch
            states = np.array([val[0] for val in batch2])
            # extract next states from the batch
            next_states = np.array([val[3] for val in batch2])

            q_s_a = self._Model2.predict_batch(states)
            q_s_a_d = self._Model2.predict_batch(next_states)

            x = np.zeros((len(batch2), self._num_states23))
            y = np.zeros((len(batch2), self._num_actions23))

            for i, b in enumerate(batch2):
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q

            self._Model2.train_batch(x, y)  # train the NN

        if len(batch3) > 0:  # if the memory is full enough
            states = np.array([val[0] for val in batch3])  # extract states from the batch
            next_states = np.array([val[3] for val in batch3])  # extract next states from the batch

            # prediction
            q_s_a = self._Model3.predict_batch(states)  # predict Q(state), for every sample
            q_s_a_d = self._Model3.predict_batch(next_states)  # predict Q(next_state), for every sample

            # setup training arrays
            x = np.zeros((len(batch3), self._num_states23))
            y = np.zeros((len(batch3), self._num_actions23))

            for i, b in enumerate(batch3):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value

            self._Model3.train_batch(x, y)  # train the NN

        if len(batch4) > 0:  # if the memory is full enough
            # extract states from the batch
            states = np.array([val[0] for val in batch4])
            # extract next states from the batch
            next_states = np.array([val[3] for val in batch4])

            # prediction
            # predict Q(state), for every sample
            q_s_a = self._Model4.predict_batch(states)
            # predict Q(next_state), for every sample
            q_s_a_d = self._Model4.predict_batch(next_states)

            # setup training arrays
            x = np.zeros((len(batch4), self._num_states4))
            y = np.zeros((len(batch4), self._num_actions4))

            for i, b in enumerate(batch4):
                # extract data from one sample
                state, action, reward, _ = b[0], b[1], b[2], b[3]
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[action] = reward + self._gamma * \
                    np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                # Q(state) that includes the updated action value
                y[i] = current_q

            self._Model4.train_batch(x, y)  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store1.append(self._sum_neg_reward1)  # how much negative reward in this episode
        self._cumulative_wait_store1.append(self._sum_waiting_time1)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store1.append(self._sum_queue_length1 / self._max_steps)  # average number of queued cars per step, in this episode

        self._reward_store2.append(self._sum_neg_reward2)
        self._cumulative_wait_store2.append(self._sum_waiting_time2)
        self._avg_queue_length_store2.append(self._sum_queue_length2 / self._max_steps)

        self._reward_store3.append(self._sum_neg_reward3)
        self._cumulative_wait_store3.append(self._sum_waiting_time3)
        self._avg_queue_length_store3.append(self._sum_queue_length3 / self._max_steps)

        self._reward_store4.append(self._sum_neg_reward4)
        self._cumulative_wait_store4.append(self._sum_waiting_time4)
        self._avg_queue_length_store4.append(self._sum_queue_length4 / self._max_steps)

        self._reward_store.append(self._sum_neg_reward1+self._sum_neg_reward2+self._sum_neg_reward3+self._sum_neg_reward4)
        self._cumulative_wait_store.append(self._sum_waiting_time1+self._sum_waiting_time2+self._sum_waiting_time3+self._sum_waiting_time4)
        self._avg_queue_length_store.append((self._sum_queue_length1+self._sum_queue_length2+self._sum_queue_length3+self._sum_queue_length4) / self._max_steps)
