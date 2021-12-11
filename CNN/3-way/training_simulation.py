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
    def __init__(self, Model, Memory, TrafficGen, sumo_cmd, gamma, max_steps, green_duration, yellow_duration, num_states, num_actions, training_epochs):
        self._Model = Model
        self._Memory = Memory
        self._TrafficGen = TrafficGen
        self._gamma = gamma
        self._step = 0
        self._sumo_cmd = sumo_cmd
        self._max_steps = max_steps  # 5400
        self._green_duration = green_duration
        self._yellow_duration = yellow_duration
        self._num_states = num_states   #80
        self._num_actions = num_actions #3
        self._reward_store = []
        self._cumulative_wait_store = []
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
        self._waiting_times = {}
        self._sum_neg_reward = 0
        self._sum_queue_length = 0
        self._sum_waiting_time = 0
        old_total_wait = 0
        old_state = -1
        old_action = -1
        c = 0

        while self._step < self._max_steps:

            # get current state of the intersection
            current_state = self._get_state()
            #t=np.array(current_state)
            #print(t.shape)
            # calculate reward of previous action: (change in cumulative waiting time between actions)
            # waiting time = seconds waited by a car since the spawn in the environment, cumulated for every car in incoming lanes
            current_total_wait = self._collect_waiting_times()
            reward = old_total_wait - current_total_wait

            # saving the data into the memory
            if self._step != 0:
                self._Memory.add_sample((old_state, old_action, reward, current_state))

            # choose the light phase to activate, based on the current state of the intersection
            action = self._choose_action(current_state, epsilon)
            # if the chosen phase is different from the last phase, activate the yellow phase
            if self._step != 0 and old_action != action:
                self._set_yellow_phase(old_action)
                self._simulate(self._yellow_duration)

            # execute the phase selected before
            self._set_green_phase(action)
            self._simulate(self._green_duration)

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_total_wait = current_total_wait

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                self._sum_neg_reward += reward

        self._save_episode_stats()
        print("Total reward:", self._sum_neg_reward, "- Epsilon:", round(epsilon, 2))
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
            queue_length = self._get_queue_length()
            self._sum_queue_length += queue_length
            self._sum_waiting_time += queue_length # 1 step while wating in queue means 1 second waited, for each car, therefore queue_lenght == waited_seconds


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


    def _choose_action(self, state, epsilon):
        """
        Decide wheter to perform an explorative or exploitative action, according to an epsilon-greedy policy
        """
        if random.random() < epsilon:
            return random.randint(0, self._num_actions - 1) # random action
        else:
            return np.argmax(self._Model.predict_one(state)) # the best action given the current state


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
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] - offset)) / cellLength)
            if(ind < 10):
                positionMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[3 - traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        for v in vehicles_road2:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[0] + offset)) / cellLength)
            if(ind < 10):
                positionMatrix[4 + traci.vehicle.getLaneIndex(v)][9 - ind] = 1
                velocityMatrix[4 + traci.vehicle.getLaneIndex(v)][9 - ind] = traci.vehicle.getSpeed(v) / speedLimit

        junctionPosition = traci.junction.getPosition('TL')[1]
        for v in vehicles_road3:
            ind = int(abs((junctionPosition - traci.vehicle.getPosition(v)[1] - offset)) / cellLength)
            if(ind < 10):
                positionMatrix[8 + 3 - traci.vehicle.getLaneIndex(v)][ind] = 1
                velocityMatrix[8 + 3 - traci.vehicle.getLaneIndex(v)][ind] = traci.vehicle.getSpeed(v) / speedLimit

        position = np.array(positionMatrix)
        position = position.reshape(1, 12, 10, 1)

        velocity = np.array(velocityMatrix)
        velocity = velocity.reshape(1, 12, 10, 1)

        return [position, velocity]

    def _replay(self):
        """
        Retrieve a group of samples from the memory and for each of them update the learning equation, then train
        """
        batch = self._Memory.get_samples(self._Model.batch_size)

        if len(batch) > 0:  # if the memory is full enough
            states = [val[0] for val in batch]  # extract states from the batch
            next_states = [val[3] for val in batch]  # extract next states from the batch

            q_s_a=[]
            q_s_a_d=[]
            # prediction
            for a in states:
              q_s_a.append(self._Model.predict_batch(a))  # predict Q(state), for every sample

            for b in next_states:
              q_s_a_d.append(self._Model.predict_batch(b))  # predict Q(next_state), for every sample

            # setup training arrays
            '''arr=[]
            for i in range(16):
              arr.append([])
              for j in range(10):
                arr.append(10)
                '''
            x=[]
            for i in range(len(batch)):
              p=[]
              v=[]
              a=[]
              for j in range(12):
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
            for i in range(len(batch)):
              b=[]
              for j in range(3):
                b.append(0)
              y.append(b)

            for i, b in enumerate(batch):
                state, action, reward, _ = b[0], b[1], b[2], b[3]  # extract data from one sample
                current_q = q_s_a[i]  # get the Q(state) predicted before
                current_q[0][action] = reward + self._gamma * np.amax(q_s_a_d[i])  # update Q(state, action)
                x[i] = state
                y[i] = current_q  # Q(state) that includes the updated action value
            for i in range(len(batch)):
                self._Model.train_batch(x[i], y[i])  # train the NN


    def _save_episode_stats(self):
        """
        Save the stats of the episode to plot the graphs at the end of the session
        """
        self._reward_store.append(self._sum_neg_reward)  # how much negative reward in this episode
        self._cumulative_wait_store.append(self._sum_waiting_time)  # total number of seconds waited by cars in this episode
        self._avg_queue_length_store.append(self._sum_queue_length / self._max_steps)  # average number of queued cars per step, in this episode


    @property
    def reward_store(self):
        return self._reward_store


    @property
    def cumulative_wait_store(self):
        return self._cumulative_wait_store

    @property
    def avg_queue_length_store(self):
        return self._avg_queue_length_store
