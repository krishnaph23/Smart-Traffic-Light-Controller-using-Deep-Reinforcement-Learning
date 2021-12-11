from __future__ import absolute_import
from __future__ import print_function

import os
from shutil import copyfile

from testing_simulation import Simulation
from generator import TrafficGenerator
from model import TestModel
from visualization import Visualization
from utils import import_test_configuration, set_sumo, set_test_path


if __name__ == "__main__":

    config = import_test_configuration(config_file='testing_settings.ini')
    sumo_cmd = set_sumo(config['gui'], config['sumocfg_file_name'], config['max_steps'])
    model_path, plot_path = set_test_path(config['models_path_name'], config['model_to_test'])
    
    Model1 = TestModel(
        '1',
        input_dim=40,
        model_path=model_path
    )

    Model2 = TestModel(
        '2',
        input_dim=30,
        model_path=model_path
    )

    Model3 = TestModel(
        '3',
        input_dim=30,
        model_path=model_path
    )

    Model4 = TestModel(
        '4',
        input_dim=50,
        model_path=model_path
    )

    TrafficGen = TrafficGenerator(
        config['max_steps'], 
        config['n_cars_generated'],
        config['n_low'],
        config['n_high']
    )

    Visualization = Visualization(
        plot_path, 
        dpi=96
    )
        
    Simulation = Simulation(
        Model1,
        Model2,
        Model3,
        Model4,
        TrafficGen,
        sumo_cmd,
        config['max_steps'],
        config['green_duration'],
        config['yellow_duration']
    )

    print('\n----- Test episode')
    simulation_time = Simulation.run(config['episode_seed'])  # run the simulation
    print('Simulation time:', simulation_time, 's')

    print("----- Testing info saved at:", plot_path)

    copyfile(src='testing_settings.ini', dst=os.path.join(plot_path, 'testing_settings.ini'))

    # Visualization.save_data_and_plot(data=Simulation._reward_episode1, filename='reward1', xlabel='Action step', ylabel='Reward')
    # Visualization.save_data_and_plot(data=Simulation._queue_length_episode1, filename='queue1', xlabel='Step', ylabel='Queue lenght (vehicles)')

    # Visualization.save_data_and_plot(data=Simulation._reward_episode2, filename='reward2', xlabel='Action step', ylabel='Reward')
    # Visualization.save_data_and_plot(data=Simulation._queue_length_episode2, filename='queue2', xlabel='Step', ylabel='Queue lenght (vehicles)')

    # Visualization.save_data_and_plot(data=Simulation._reward_episode3, filename='reward3', xlabel='Action step', ylabel='Reward')
    # Visualization.save_data_and_plot(data=Simulation._queue_length_episode3, filename='queue3', xlabel='Step', ylabel='Queue lenght (vehicles)')

    # Visualization.save_data_and_plot(data=Simulation._reward_episode4, filename='reward4', xlabel='Action step', ylabel='Reward')
    # Visualization.save_data_and_plot(data=Simulation._queue_length_episode4, filename='queue4', xlabel='Step', ylabel='Queue lenght (vehicles)')

    Visualization.save_data_and_plot(data=Simulation._reward_episode, filename='reward', xlabel='Action step', ylabel='Reward')
    Visualization.save_data_and_plot(data=Simulation._queue_length_episode, filename='queue', xlabel='Step', ylabel='Queue lenght (vehicles)')
