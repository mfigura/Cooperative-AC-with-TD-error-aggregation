import numpy as np
import gym
from gym import spaces

class Synthetic_example(gym.Env):
    """
    Synthetic example
    This is a synthetic example with local coupling in the dynamics and rewards. Each agent can visit
    two states (0 or 1) and chooses an action (0 or 1).
    ARGUMENTS:  n_agents: total number of agents
                global_obs: global observability (True or False)
    """
    metadata = {'render.modes': ['console']}
    def __init__(self, n_agents = 1,global_obs=False):
        self.n_agents = n_agents
        self.n_agent_actions = 2
        self.n_agent_states = 2
        self.global_obs = global_obs
        self.action_space = gym.spaces.MultiDiscrete([self.n_agent_actions for _ in range(self.n_agents)])
        self.observation_space = gym.spaces.MultiDiscrete([self.n_agent_states for _ in range(self.n_agents)])
        self.observation_dim = self.n_agents if self.global_obs else 1

        self.reset()

    def _state_and_reward(self,state,action):
        '''
        Computes a new state given the current state and action
        Arguments: state and action
        Returns: new local state, local reward
        '''
        prob = np.zeros(self.n_agents)
        reward = np.zeros(self.n_agents)
        for i in range(self.n_agents):
            prob[i] = (np.sum(state)+np.sum(action))/(2*self.n_agents)
        new_state = np.random.binomial(n=1,p=prob*action)
        reward[0] = prob[0]

        return new_state,reward

    def reset(self):
        '''Resets the environment'''
        self.observations = self.observation_space.sample()
        obs = np.tile(self.observations,(self.n_agents,1)) if self.global_obs else np.array(self.observations).reshape(self.n_agents,1)

        return obs

    def step(self, global_action):
        '''
        Makes a transition to a new state and evaluates all rewards
        Arguments: global action
        '''
        new_obs,rewards = self._state_and_reward(self.observations,global_action)
        #print(self.observations,global_action,rewards,new_obs)
        self.observations = np.array(new_obs)
        obs = np.tile(new_obs,(self.n_agents,1)) if self.global_obs else np.array(new_obs).reshape(self.n_agents,1)

        self.done = False
        self.info = None

        return obs,rewards,self.done,self.info

    def close(self):
        pass
