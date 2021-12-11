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

        p_high = self._n_high/self._n_cars_generated
        
        # produce the file for cars generation, one car per line
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

            <route id="A_B" edges="A2TL1 TL12B"/>
            <route id="A_C" edges="A2TL1 TL12TL2 TL22C"/>
            <route id="A_D" edges="A2TL1 TL12TL2 TL22D"/>
            <route id="A_E" edges="A2TL1 TL12TL2 TL22E"/>
            <route id="A_F" edges="A2TL1 TL12F"/>

            <route id="B_A" edges="B2TL1 TL12A"/>
            <route id="B_C" edges="B2TL1 TL12TL2 TL22C"/>
            <route id="B_D" edges="B2TL1 TL12TL2 TL22D"/>
            <route id="B_E" edges="B2TL1 TL12TL2 TL22E"/>
            <route id="B_F" edges="B2TL1 TL12F"/>

            <route id="C_A" edges="C2TL2 TL22TL1 TL12A"/>
            <route id="C_B" edges="C2TL2 TL22TL1 TL12B"/>
            <route id="C_D" edges="C2TL2 TL22D"/>
            <route id="C_E" edges="C2TL2 TL22E"/>
            <route id="C_F" edges="C2TL2 TL22TL1 TL12F"/>

            <route id="D_A" edges="D2TL2 TL22TL1 TL12A"/>
            <route id="D_B" edges="D2TL2 TL22TL1 TL12B"/>
            <route id="D_C" edges="D2TL2 TL22C"/>
            <route id="D_E" edges="D2TL2 TL22E"/>
            <route id="D_F" edges="D2TL2 TL22TL1 TL12F"/>

            <route id="E_A" edges="E2TL2 TL22TL1 TL12A"/>
            <route id="E_B" edges="E2TL2 TL22TL1 TL12B"/>
            <route id="E_C" edges="E2TL2 TL22C"/>
            <route id="E_D" edges="E2TL2 TL22D"/>
            <route id="E_F" edges="E2TL2 TL22TL1 TL12F"/>

            <route id="F_A" edges="F2TL1 TL12A"/>
            <route id="F_B" edges="F2TL1 TL12B"/>
            <route id="F_C" edges="F2TL1 TL12TL2 TL22C"/>
            <route id="F_D" edges="F2TL1 TL12TL2 TL22D"/>
            <route id="F_E" edges="F2TL1 TL12TL2 TL22E"/>""", file=routes)

            for car_counter, step in enumerate(car_gen_steps):
                straight_or_turn = np.random.uniform()
                low_or_high = np.random.uniform()
                if low_or_high <= p_high:
                    veh_type = "high_priority"
                    pr='P'
                else:
                    veh_type = "low_priority"
                    pr=''

                route = np.random.randint(1, 31)  # choose a random source & destination
                if route == 1:
                    print('    <vehicle id="%sA_B%i" type="%s" route="A_B" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 2:
                    print('    <vehicle id="%sA_C%i" type="%s" route="A_C" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 3:
                    print('    <vehicle id="%sA_D%i" type="%s" route="A_D" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 4:
                    print('    <vehicle id="%sA_E%i" type="%s" route="A_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 5:
                    print('    <vehicle id="%sA_F%i" type="%s" route="A_F" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

                elif route == 6:
                    print('    <vehicle id="%sB_A%i" type="%s" route="B_A" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 7:
                    print('    <vehicle id="%sB_C%i" type="%s" route="B_C" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 8:
                    print('    <vehicle id="%sB_D%i" type="%s" route="B_D" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 9:
                    print('    <vehicle id="%sB_E%i" type="%s" route="B_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 10:
                    print('    <vehicle id="%sB_F%i" type="%s" route="B_F" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

                elif route == 11:
                    print('    <vehicle id="%sC_A%i" type="%s" route="C_A" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 12:
                    print('    <vehicle id="%sC_B%i" type="%s" route="C_B" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 13:
                    print('    <vehicle id="%sC_D%i" type="%s" route="C_D" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 14:
                    print('    <vehicle id="%sC_E%i" type="%s" route="C_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 15:
                    print('    <vehicle id="%sC_F%i" type="%s" route="C_F" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

                elif route == 16:
                    print('    <vehicle id="%sD_A%i" type="%s" route="D_A" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 17:
                    print('    <vehicle id="%sD_B%i" type="%s" route="D_B" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 18:
                    print('    <vehicle id="%sD_C%i" type="%s" route="D_C" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 19:
                    print('    <vehicle id="%sD_E%i" type="%s" route="D_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 20:
                    print('    <vehicle id="%sD_F%i" type="%s" route="D_F" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

                elif route == 21:
                    print('    <vehicle id="%sE_A%i" type="%s" route="E_A" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 22:
                    print('    <vehicle id="%sE_B%i" type="%s" route="E_B" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 23:
                    print('    <vehicle id="%sE_C%i" type="%s" route="E_C" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 24:
                    print('    <vehicle id="%sE_D%i" type="%s" route="E_D" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 25:
                    print('    <vehicle id="%sE_F%i" type="%s" route="E_F" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

                elif route == 26:
                    print('    <vehicle id="%sF_A%i" type="%s" route="F_A" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 27:
                    print('    <vehicle id="%sF_B%i" type="%s" route="F_B" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 28:
                    print('    <vehicle id="%sF_C%i" type="%s" route="F_C" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 29:
                    print('    <vehicle id="%sF_D%i" type="%s" route="F_D" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)
                elif route == 30:
                    print('    <vehicle id="%sF_E%i" type="%s" route="F_E" depart="%s" departLane="random" departSpeed="10" />' % (pr, car_counter, veh_type, step), file=routes)

            print("</routes>", file=routes)


# call= TrafficGenerator(5400,1000,10,900)
# call.generate_routefile(0)
