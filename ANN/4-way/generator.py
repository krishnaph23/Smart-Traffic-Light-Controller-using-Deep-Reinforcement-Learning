import numpy as np
import math

class TrafficGenerator:
    def __init__(self, max_steps, n_cars_generated, n_low, n_high):
        self._n_cars_generated = n_cars_generated  # how many cars per episode
        self._max_steps = max_steps
        self._n_low = n_low
        self._n_high = n_high

    def generate_routefile(self, seed):
        """
        Generation of the route of every car for one episode
        """
        np.random.seed(seed)  # make tests reproducible

        # the generation of cars is distributed according to a weibull distribution
        timings = np.random.weibull(2, self._n_cars_generated)
        timings = np.sort(timings)

        # reshape the distribution to fit the interval 0:max_steps
        car_gen_steps = []
        min_old = math.floor(timings[1])
        max_old = math.ceil(timings[-1])
        min_new = 0
        max_new = self._max_steps
        for value in timings:
            car_gen_steps = np.append(car_gen_steps, ((max_new - min_new) / (max_old - min_old)) * (value - max_old) + max_new)

        car_gen_steps = np.rint(car_gen_steps)  # round every value to int -> effective steps when a car will be generated

        # produce the file for cars generation, one car per line
        # p_low = self._n_low/self._n_cars_generated
        
        p_high = self._n_high/self._n_cars_generated
        
        with open("intersection/episode_routes.rou.xml", "w") as routes:
            print("""<routes>
            <vTypeDistribution id="low_priority">
                <vType id="standard_car" vClass="passenger" guiShape="passenger" speedDev="0.2" latAlignment="arbitrary" sigma="0.5" probability="0.5" />
                <vType id="bike" vClass="motorcycle" guiShape="motorcycle" speedDev="0.2" latAlignment="arbitrary" sigma="0.5" probability="0.1"/>
                <vType id="bus" vClass="bus" guiShape="truck" speedDev="0.2" latAlignment="arbitrary" sigma="0.5" probability="0.2"/>
                <vType id="taxi" vClass="taxi" guiShape="passenger/sedan" speedDev="0.2" latAlignment="arbitrary" sigma="0.5" color = "255,0,0" probability="0.2"/>
            </vTypeDistribution>
        
            <vTypeDistribution id="high_priority">
                <vType id="amb" vClass="emergency" guiShape="emergency" speedDev="0.2" latAlignment="arbitrary" probability="0.4"><param key="has.bluelight.device" value="true"/></vType>
                <vType id="pol" vClass="authority" guiShape="police" speedDev="0.4" latAlignment="arbitrary" probability="0.4"><param key="has.bluelight.device" value="true"/></vType>
                <vType id="fire" vClass="emergency" guiShape="firebrigade" speedDev="0.4" latAlignment="arbitrary" probability="0.2"><param key="has.bluelight.device" value="true"/></vType>
            </vTypeDistribution>
            
            <route id="W_N" edges="W2TL TL2N"/>
            <route id="W_E" edges="W2TL TL2E"/>
            <route id="W_S" edges="W2TL TL2S"/>
            <route id="N_W" edges="N2TL TL2W"/>
            <route id="N_E" edges="N2TL TL2E"/>
            <route id="N_S" edges="N2TL TL2S"/>
            <route id="E_W" edges="E2TL TL2W"/>
            <route id="E_N" edges="E2TL TL2N"/>
            <route id="E_S" edges="E2TL TL2S"/>
            <route id="S_W" edges="S2TL TL2W"/>
            <route id="S_N" edges="S2TL TL2N"/>
            <route id="S_E" edges="S2TL TL2E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                low_or_high = np.random.uniform()
                if low_or_high <= p_high:
                    veh_type = "high_priority"
                    pr='P'
                else:
                    veh_type = "low_priority"
                    pr=''
                
                route = np.random.randint(1, 13)  # choose a random source & destination
                if route == 1:
                    print('    <vehicle id="%sN_W_%i" type="%s" route="N_W" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 2:
                    print('    <vehicle id="%sN_S_%i" type="%s" route="N_S" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 3:
                    print('    <vehicle id="%sN_E_%i" type="%s" route="N_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 4:
                    print('    <vehicle id="%sW_E_%i" type="%s" route="W_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 5:
                    print('    <vehicle id="%sW_S_%i" type="%s" route="W_S" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 6:
                    print('    <vehicle id="%sW_N_%i" type="%s" route="W_N" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 7:
                    print('    <vehicle id="%sE_W_%i" type="%s" route="E_W" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 8:
                    print('    <vehicle id="%sE_N_%i" type="%s" route="E_N" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 9:
                    print('    <vehicle id="%sE_S_%i" type="%s" route="E_S" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 10:
                    print('    <vehicle id="%sS_N_%i" type="%s" route="S_N" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 11:
                    print('    <vehicle id="%sS_W_%i" type="%s" route="S_W" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                else:
                    print('    <vehicle id="%sS_E_%i" type="%s" route="S_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
            

            print("</routes>", file=routes)


call= TrafficGenerator(5400,1000,10,900)
call.generate_routefile(0)
