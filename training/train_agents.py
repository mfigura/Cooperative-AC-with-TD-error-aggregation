import numpy as np
import math
import gym
from gym import spaces
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Input, Model, Sequential, layers
import pandas as pd

'This file contains a function for training networked AC agents in custom multi-agent gym environments.'

def train_batch(env,agents,args):
    '''
    FUNCTION train_batch() - training a cooperative network of decentralized actor-critic agents subject to communication delays in training.
    The agents apply actions sampled from the actor network and update their actor and critic networks in batches.
    ARGUMENTS: gym environment, list of decentralized AC agents, training parameters
    RETURNS: trained agents, simulation data
    '''
    paths = []
    n_agents, n_observations, n_actions = env.n_agents, env.observation_dim, env.n_agent_actions
    gamma = args['gamma']
    in_nodes = args['in_nodes']
    update_frequency = args['update_frequency']
    communication_frequency = args['communication_frequency']
    buffer_size = agents[0].buffer_size
    n_episodes = args['n_episodes']
    max_ep_len = args['max_ep_len']
    cum_avg = 0

    observations = np.zeros((n_agents,n_episodes,max_ep_len+1,n_observations))
    actions = np.zeros((n_agents,n_episodes,max_ep_len),dtype=int)
    rewards = np.zeros((n_agents,n_episodes,max_ep_len,1))
    TDE_matrices = np.zeros((n_agents,max(buffer_size,1),n_agents,update_frequency*max_ep_len,1))

    obs_check = np.array([[0,0],[0,1],[1,0],[1,1]])
    writer = tf.summary.create_file_writer(logdir = args['summary_dir'])

    t = 0
    while t < n_episodes:

        'BEGINNING OF A TRAINING EPISODE'

        #----------------------------------------------------------------------------------------------

        'EPISODE SIMULATION'
        j,ep_returns = 0,0
        observations[:,t,0] = env.reset()
        est_returns = [agents[node].critic(observations[node,t,0].reshape(1,-1)).numpy()[0][0] for node in range(n_agents)]
        while j < max_ep_len:
            for node in range(n_agents):
                actions[node,t,j] = agents[node].get_action(observations[node,t,j].reshape(1,-1))
            observations[:,t,j+1], rewards[:,t,j,0], done, _ = env.step(actions[:,t,j])
            ep_returns += rewards[:,t,j]*(gamma**j)
            j += 1

        #print("Observations: "+str(observations))
        #print("Actions: "+str(actions))
        #print("Rewards: "+str(rewards))

        'SUMMARY OF THE TRAINING EPISODE'
        mean_true_returns = np.mean(ep_returns)
        with writer.as_default():
            tf.summary.scalar("true episode team-average returns",mean_true_returns, step = t)
            writer.flush()
        print('| Episode: {} | Est. returns: {} | Returns: {} | Mean returns: {} '.format(t,est_returns,ep_returns.ravel(),mean_true_returns))
        path = {"True_team_returns":mean_true_returns,"Estimated_team_returns":est_returns}
        paths.append(path)
        cum_avg += mean_true_returns/update_frequency
        #------------------------------------------------------------------------------------------------

        'RECEIVE AND AGGREGATE TD ERROR MATRICES'
        if ((t+1) % communication_frequency) == 0:
            print("Receiving messages...")
            for node in range(n_agents):
                agents[node].TDE_matrix_aggregate(TDE_matrices[in_nodes[node]])
            for node in range(n_agents):
                TDE_matrices[node] = agents[node].TDE_matrix

        #------------------------------------------------------------------------------------------------

        'ALGORITHM UPDATES'
        if ((t+1) % update_frequency) == 0:
            #print("Action probabilities: "+str(agents[1].actor.predict(obs_check)))
            print("Cumulated average: "+str(cum_avg))
            cum_avg = 0
            print("Updating algorithm...")
            states = observations[:,t+1-update_frequency:t+1,:-1].reshape(n_agents,-1,n_observations)
            new_states = observations[:,t+1-update_frequency:t+1,1:].reshape(n_agents,-1,n_observations)
            acts = actions[:,t+1-update_frequency:t+1].reshape(n_agents,-1,1)
            rews = rewards[:,t+1-update_frequency:t+1].reshape(n_agents,-1,1)
            for node in range(n_agents):
                'UPDATE STATE AND ACTION BUFFERS'
                agents[node].update_buffers(states[node],acts[node])
                'CRITIC UPDATE'
                agents[node].critic_update(states[node],new_states[node],rews[node])
                'TD ERROR MATRIX UPDATE'
                TDE_matrices[node] = agents[node].TDE_matrix_update(states[node],new_states[node],rews[node])
                'ACTOR UPDATE'
                agents[node].actor_update()

        #-----------------------------------------------------------------------------------------------
        t += 1

    sim_data = pd.DataFrame.from_dict(paths)
    return agents,sim_data
