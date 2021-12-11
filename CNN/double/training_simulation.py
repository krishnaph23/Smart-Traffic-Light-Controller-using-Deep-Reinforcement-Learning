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

        self._sum_queue_length1 = 0
        self._sum_queue_length2 = 0

        self._sum_waiting_time1 = 0
        self._sum_waiting_time2 = 0

        old_total_wait1 = 0
        old_total_wait2 = 0

        old_state1 = -1
        old_action1 = -1
        old_state2 = -1
        old_action2 = -1

        while self._step < self._max_steps:
            current_state1 = self._get_state(1)
            current_state2 = self._get_state(2)
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

    def _get_state(self,inter):
        """
        Retrieve the state of the intersection from sumo, in the form of cell occupancy
        """
        positionMatrix = []
        velocityMatrix = []

        cellLength = 7
        offset = 11
        speedLimit = 14

        if inter==1:
            junctionPosition = traci.junction.getPosition('TL1')[0]
            vehicles_road1 = traci.edge.getLastStepVehicleIDs('A2TL1')
            vehicles_road2 = traci.edge.getLastStepVehicleIDs('B2TL1')
            vehicles_road3 = traci.edge.getLastStepVehicleIDs('TL22TL1')
            vehicles_road4 = traci.edge.getLastStepVehicleIDs('F2TL1')
            for i in range(16):
                positionMatrix.append([])
                velocityMatrix.append([])
                for j in range(10):
                    positionMatrix[i].append(0)
                    velocityMatrix[i].append(0)

            for v in vehicles_road1:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[3 - traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                    velocityMatrix[3 - traci.vehicle.getLaneIndex(
                        v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            for v in vehicles_road2:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[4 + traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

            junctionPosition = traci.junction.getPosition('TL1')[1]
            for v in vehicles_road3:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[8 + 3 -
                                traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                    velocityMatrix[8 + 3 - traci.vehicle.getLaneIndex(
                        v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            for v in vehicles_road4:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[12 + traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

            position = np.array(positionMatrix)
            position = position.reshape(1, 16, 10, 1)

            velocity = np.array(velocityMatrix)
            velocity = velocity.reshape(1, 16, 10, 1)
        
        elif inter==2:
            junctionPosition = traci.junction.getPosition('TL2')[0]
            vehicles_road1 = traci.edge.getLastStepVehicleIDs('TL12TL2')
            vehicles_road2 = traci.edge.getLastStepVehicleIDs('C2TL2')
            vehicles_road3 = traci.edge.getLastStepVehicleIDs('D2TL2')
            vehicles_road4 = traci.edge.getLastStepVehicleIDs('E2TL2')
            for i in range(16):
                positionMatrix.append([])
                velocityMatrix.append([])
                for j in range(10):
                    positionMatrix[i].append(0)
                    velocityMatrix[i].append(0)

            for v in vehicles_road1:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[3 - traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                    velocityMatrix[3 - traci.vehicle.getLaneIndex(
                        v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            for v in vehicles_road2:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[4 + traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[4 + traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

            junctionPosition = traci.junction.getPosition('TL2')[1]
            for v in vehicles_road3:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[8 + 3 -
                                traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                    velocityMatrix[8 + 3 - traci.vehicle.getLaneIndex(
                        v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

            for v in vehicles_road4:
                ind = int(
                    abs((junctionPosition - traci.vehicle.getPosition(v)[1] + offset)) / cellLength)
                if(ind < 10):
                    positionMatrix[12 + traci.vehicle.getLaneIndex(v)][ind] = 1
                    velocityMatrix[12 + traci.vehicle.getLaneIndex(
                        v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

            position = np.array(positionMatrix)
            position = position.reshape(1, 16, 10, 1)

            velocity = np.array(velocityMatrix)
            velocity = velocity.reshape(1, 16, 10, 1)
        return [position, velocity]
  

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch1 = self._Memory1.get_samples(self._Model1.batch_size)
        batch2 = self._Memory2.get_samples(self._Model2.batch_size)

        if len(batch1) > 0:
            states = [val[0] for val in batch1]  # extract states from the batch
            next_states = [val[3] for val in batch1]  # extract next states from the batch

            q_s_a=[]
            q_s_a_d=[]
            # prediction
            for a in states:
              q_s_a.append(self._Model1.predict_batch(a))  # predict Q(state), for every sample
    
            for b in next_states:  
              q_s_a_d.append(self._Model1.predict_batch(b))  # predict Q(next_state), for every sample

            # setup training arrays
            '''arr=[]
            for i in range(16):
              arr.append([])
              for j in range(10):
                arr.append(10)
                '''
            x=[]
            for i in range(len(batch1)):
              p=[]
              v=[]
              a=[]
              for j in range(16):
                p.append([])
                v.append([])
                for j in range(10):
                  p.append(0)
                  v.append(0)
                a.append(p)
                a.append(v)
              x.append(a)
            
            '''x=[]
            t=np.zeros((16,10,1))
            t1=[]
            t1.append(t)
            t1.append(t)
            for i in range(len(batch)):
              x.append(t1)
            '''
            #x = np.zeros((len(batch), 2,1,16,10,1))
            #y = np.zeros((len(batch), self._num_actions))
            y=[]
            for i in range(len(batch1)):
              b=[]
              for j in range(4):
                b.append(0)
              y.append(b)

            for i, b in enumerate(batch1):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[0][action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            for i in range(len(batch1)):
                self._Model1.train_batch(x[i], y[i])  # train the NN

        if len(batch2) > 0:
            states = [val[0] for val in batch2]  # extract states from the batch
            next_states = [val[3] for val in batch2]  # extract next states from the batch

            q_s_a=[]
            q_s_a_d=[]
            # prediction
            for a in states:
              q_s_a.append(self._Model2.predict_batch(a))  # predict Q(state), for every sample
    
            for b in next_states:  
              q_s_a_d.append(self._Model2.predict_batch(b))  # predict Q(next_state), for every sample

            # setup training arrays
            '''arr=[]
            for i in range(16):
              arr.append([])
              for j in range(10):
                arr.append(10)
                '''
            x=[]
            for i in range(len(batch2)):
              p=[]
              v=[]
              a=[]
              for j in range(16):
                p.append([])
                v.append([])
                for j in range(10):
                  p.append(0)
                  v.append(0)
                a.append(p)
                a.append(v)
              x.append(a)
            
            '''x=[]
            t=np.zeros((16,10,1))
            t1=[]
            t1.append(t)
            t1.append(t)
            for i in range(len(batch)):
              x.append(t1)
            '''
            #x = np.zeros((len(batch), 2,1,16,10,1))
            #y = np.zeros((len(batch), self._num_actions))
            y=[]
            for i in range(len(batch2)):
              b=[]
              for j in range(4):
                b.append(0)
              y.append(b)

            for i, b in enumerate(batch2):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[0][action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            for i in range(len(batch2)):
                self._Model2.train_batch(x[i], y[i])  # train the NN

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
