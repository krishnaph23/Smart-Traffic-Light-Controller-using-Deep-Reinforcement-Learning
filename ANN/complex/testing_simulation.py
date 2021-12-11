import traci
import numpy as np
import random
import timeit
import os

PHASE_N_GREEN = 0  # action 0 code 00
PHASE_N_YELLOW = 1
PHASE_E_GREEN = 2  # action 1 code 01
PHASE_E_YELLOW = 3
PHASE_S_GREEN = 4  # action 2 code 10
PHASE_S_YELLOW = 5  # action*2 +1
PHASE_W_GREEN = 6  # action 3 code 11
PHASE_W_YELLOW = 7  # action*2 + 1
PHASE_J_GREEN = 8  # action 3 code 11
PHASE_J_YELLOW = 9  # action*2 + 1


class Simulation:
    def __init__(self, Model1, Model2, Model3, Model4, TrafficGen, sumo_cmd, max_steps, green_duration, yellow_duration):
        self._Model1 = Model1
        self._Model2 = Model2
        self._Model3 = Model3
        self._Model4 = Model4

        self._TrafficGen = TrafficGen
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states1 = 40  # 80
        self._num_states23 = 30  # 80
        self._num_states4 = 50  # 80

        self._reward_episode1 = []
        self._reward_episode2 = []
        self._reward_episode3 = []
        self._reward_episode4 = []

        self._reward_episode = []

        self._queue_length_episode1 = []
        self._queue_length_episode2 = []
        self._queue_length_episode3 = []
        self._queue_length_episode4 = []

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

        self._waiting_times1 = {}
        self._waiting_times2 = {}
        self._waiting_times3 = {}
        self._waiting_times4 = {}

        old_total_wait1 = 0
        old_total_wait2 = 0
        old_total_wait3 = 0
        old_total_wait4 = 0

        old_action1 = -1  # dummy init
        old_action2 = -1  # dummy init
        old_action3 = -1  # dummy init
        old_action4 = -1  # dummy init

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state1, current_state2, current_state3, current_state4 = self._get_state()
            current_total_wait1, current_total_wait2, current_total_wait3, current_total_wait4 = self._collect_waiting_times()

            reward1 = old_total_wait1 - current_total_wait1
            reward2 = old_total_wait2 - current_total_wait2
            reward3 = old_total_wait3 - current_total_wait3
            reward4 = old_total_wait4 - current_total_wait4

            # choose the light phase to activate, based on the current state of the intersection

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

            # print(action2)
            # saving variables for later & accumulate reward
            old_action1 = action1
            old_action2 = action2
            old_action3 = action3
            old_action4 = action4

            old_total_wait1 = current_total_wait1
            old_total_wait2 = current_total_wait2
            old_total_wait3 = current_total_wait3
            old_total_wait4 = current_total_wait4

            self._reward_episode1.append(reward1)
            self._reward_episode2.append(reward2)
            self._reward_episode3.append(reward3)
            self._reward_episode4.append(reward4)

            self._reward_episode.append(reward1+reward2+reward3+reward4)

        #print("Total reward:", np.sum(self._reward_episode))
        traci.close()
        simulation_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time

    def _simulate(self, steps_todo):
        # do not do more steps than the maximum allowed number of steps
        if (self._step + steps_todo) >= self._max_steps:
            steps_todo = self._max_steps - self._step

        while steps_todo > 0:
            traci.simulationStep()  # simulate 1 step in sumo
            self._step += 1  # update the step counter
            steps_todo -= 1
            queue_length1, queue_length2, queue_length3, queue_length4 = self._get_queue_length()

            self._queue_length_episode1.append(queue_length1)
            self._queue_length_episode2.append(queue_length2)
            self._queue_length_episode3.append(queue_length3)
            self._queue_length_episode4.append(queue_length4)
            
            self._queue_length_episode.append(queue_length1+queue_length2+queue_length3+queue_length4)

    def _collect_waiting_times(self):
        incoming_roads1 = ["A2TL1", "B2TL1", "C2TL1", "TL22TL1"]
        incoming_roads2 = ["TL12TL2", "TL32TL2", "TL42TL2"]
        incoming_roads3 = ["G2TL3", "TL22TL3", "TL42TL3"]
        incoming_roads4 = ["D2TL4", "E2TL4", "F2TL4", "TL22TL4", "TL32TL4"]

        car_list = traci.vehicle.getIDList()
        for car_id in car_list:
            wait_time = traci.vehicle.getAccumulatedWaitingTime(car_id)
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
