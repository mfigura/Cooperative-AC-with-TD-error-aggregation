import numpy as np
import gym
import argparse
import pickle
import pandas as pd
import math
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
from environments.movers_world import Movers_World
from environments.cooperative_navigation import Cooperative_navigation
from environments.synthetic_example import Synthetic_example
from agents.decentralized_AC_agent import DAC_agent
import training.train_agents as training

'''
This is a main file, where the user selects the environment, learning hyperparameters, environment parameters,
and simulation parameters as well as neural network architecture for the actor and critic. The script triggers
a training process whose results are passed to folder Simulation_results.
'''

if __name__ == '__main__':

    '''USER-DEFINED PARAMETERS'''
    parser = argparse.ArgumentParser(description='Provide parameters for training cooperative AC agents')
    #Environment parameters
    parser.add_argument('--n_agents',help='total number of agents',type=int,default=2)
    parser.add_argument('--in_nodes',help='specify a list of neighbors that transmit values to each agent (include the index of self)',type=int,default=[[0,1],[0,1]])
    parser.add_argument('--global_observability',help='specify whether each agent makes full observation from the environment',type=bool,default=False)
    #parser.add_argument('--grid_dimension',help='Grid world dimension',type=int,default=(5,5))
    #parser.add_argument('--desired_state',help='desired state of the agents',type=int,default=np.array([[0,0],[4,0],[0,4],[4,4],[2,2]]))
    #Simulation parameters
    parser.add_argument('--n_episodes', help='number of episodes', type=int, default=100)
    parser.add_argument('--max_ep_len', help='Number of steps per episode', type=int, default=100)
    #Training parameters
    parser.add_argument('--update_frequency', help='Frequency of updates (episodes/update)', type=int, default=1)
    parser.add_argument('--communication_frequency', help='Communication frequency (episodes/update)', type=int, default=1)
    parser.add_argument('--actor_lr', help='actor learning rate',type=float, default=0.1)
    parser.add_argument('--critic_lr', help='critic learning rate',type=float, default=0.01)
    parser.add_argument('--gamma', help='discount factor', type=float, default=0.9)
    parser.add_argument('--local_policy_optimization',help='set to True if agents maximize their policy based on the data from their neighborhood (specified in in_nodes)',type=bool,default=False)
    parser.add_argument('--graph_diam',help='graph diameter',type=int,default=0)
    parser.add_argument('--n_train_epochs',help='critic training: number of epochs when TD targets are fixed and number of times the TD targets are updated',type=int,default=5)
    #Auxiliary parameters
    parser.add_argument('--scaling',help='Normalize observations',type=bool,default=False)
    parser.add_argument('--summary_dir',help='Create a directory to save simulation results', default='./simulation_results/')
    parser.add_argument('--random_seed',help='Set random seed for the random number generator',type=int,default=None)

    args = vars(parser.parse_args())
    np.random.seed(args['random_seed'])
    tf.random.set_seed(args['random_seed'])
    args['desired_state'] = np.random.randint(0,5,size=(5,2))

    #----------------------------------------------------------------------------------------------------------------------------------------
    '''CREATE ENVIRONMENT'''

    env = Synthetic_example(
                       #grid_dim = args['grid_dimension'],
                       n_agents = args['n_agents'],
                       #desired_position = args['desired_state'],
                       #scaling = args['scaling'],
                       global_obs = args['global_observability']
                      )
    #----------------------------------------------------------------------------------------------------------------------------------------
    '''CREATE AGENTS'''

    agents = []

    for node in range(args['n_agents']):

        actor = keras.Sequential([
                                    keras.layers.Dense(10, activation=keras.layers.LeakyReLU(alpha=0.3),input_shape=(env.observation_dim,)),
                                    keras.layers.Dense(10, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(env.n_agent_actions,activation='softmax')
                                  ])

        critic = keras.Sequential([
                                    keras.layers.Dense(5, activation=keras.layers.LeakyReLU(alpha=0.3),input_shape=(env.observation_dim,)),
                                    keras.layers.Dense(5, activation=keras.layers.LeakyReLU(alpha=0.3)),
                                    keras.layers.Dense(1)
                                  ])

        agents.append(DAC_agent(actor,
                                critic,
                                actor_lr = args['actor_lr'],
                                critic_lr = args['critic_lr'],
                                gamma = args['gamma'],
                                buffer_size = int(math.ceil(args['communication_frequency']*args['graph_diam']/args['update_frequency'])),
                                data_size = args['update_frequency']*args['max_ep_len'],
                                agent_index = node,
                                local_policy_optimization = args['local_policy_optimization'],
                                neighbor_node_indices = args['in_nodes'][node],
                                n_agents = env.n_agents,
                                n_agent_actions = env.n_agent_actions,
                                n_train_epochs = args['n_train_epochs']
                                )
                      )
    print(args)
    #---------------------------------------------------------------------------------------------------------------------------------------------
    '''TRAIN AGENTS'''
    trained_agents,sim_data = training.train_batch(env,agents,args)
    #----------------------------------------------------------------------------------------------------
    sim_data.to_pickle(args['summary_dir']+"sim_data.pkl")
