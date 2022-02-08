import numpy as np
import tensorflow as tf
from tensorflow import keras

class DAC_agent():
    '''
    DECENTRALIZED ACTOR-CRITIC AGENT
    This is an implementation of the decentralized actor-critic (DAC) algorithm with TD error fusion from Figura et al. (2021).
    The algorithm is a realization of temporal difference learning with one-step lookahead. It is an instance of decentralized
    cooperative learning, where each agent receives a local reward and observes the local state and action. The DAC agent seeks
    to maximize a team-average objective function. To achieve that, the agents aggregate TD errors from the network. The communication
    between the agents is assumed to be delayed such that agents receive full information from the network in k steps. The DAC agent
    employs neural networks to approximate the actor and critic.

    Our implementation assumes that the actor and critic updates are performed on separate timescales. Both the actor and critic
    updates are applied using batch stochastic gradient descent. The algorithm updates are described below.

    1) Action sampling - get_action()
    The agent samples an action at a given state from the current policy.

    2) TD error matrix aggregation - TDE_matrix_aggregate()
    This method is part of the communication of TD errors between agents. It aggregates received values and updates the local TD error matrix.

    3) Updating buffers with states, actions, and actor weights - update_buffers()
    This method stores states, actions, and actor weights in the agent's buffers. These data are important for the evaluation of delayed actor gradients.

    4) Critic update - critic_update()
    The critic is trained over a number of epochs in which the TD targets are periodically re-evaluated.

    5) TD error matrix update - TDE_matrix_update()
    The agent evaluates the team-average TD error(s) delayed by k steps. Furthermore, it updates the local TD error matrix with the most recent TD error(s).

    5) Actor update - actor_update()
    The actor updates apply gradient that are evaluated based on the data stored in buffers k steps ago.

    Final notes: The algorithm admits both a batch and online implementation.
                 Communication is required only while training, not at test time.
                 In episodic training, one can assume that the communication between agents and actor-critic updates take place at the end of an episode.

    ARGUMENTS: actor (keras model)
               critic (keras model)
               actor learning rate
               critic learning rate
               agent's index in the network
               number of networked agents
               data size = number of episodes in between two actor updates * episode length
               discount factor gamma
               buffer size
               number of epochs when TD targets are kept fixed in the critic updates (total number of training epochs = n_train_epochs**2)
    '''
    def __init__(self,actor,critic,actor_lr,critic_lr,agent_index,n_agents,n_agent_actions,data_size,neighbor_node_indices,local_policy_optimization=False,gamma=0.95,buffer_size=2,n_train_epochs=5):
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.agent_index = agent_index
        self.n_agents = n_agents
        self.n_agent_actions = n_agent_actions
        self.data_size = data_size
        self.n_train_epochs = n_train_epochs
        self.in_nodes = neighbor_node_indices
        self.local_policy_optimization = local_policy_optimization

        self.critic.compile(optimizer=keras.optimizers.SGD(learning_rate=critic_lr),loss=keras.losses.MeanSquaredError())
        self.actor.compile(optimizer=keras.optimizers.SGD(learning_rate=actor_lr),loss=keras.losses.SparseCategoricalCrossentropy())
        self.actor_weights_buffer = []
        self.state_buffer = []
        self.action_buffer = []
        self.TDE_matrix = np.zeros((max(self.buffer_size,1),self.n_agents,self.data_size,1))

    def get_action(self,state):
        '''Choose an action from the policy at the current state'''
        state = np.array(state).reshape(1,-1)
        action_prob = self.actor.predict(state)
        action_from_policy = np.random.choice(self.n_agent_actions, p = action_prob[0])

        return action_from_policy

    def TDE_matrix_aggregate(self,TDE_matrices):
        '''
        Update of the TD error matrix
        - computes the mean of nonzero entries over all neighbors (equivalent to copying values)
        - updates a local buffer with team-average TD errors
        Arguments: TD error matrices stacked in the first dimension (3D np array)
        '''
        numerator = np.sum(TDE_matrices,axis=0)
        denominator = np.maximum(np.count_nonzero(TDE_matrices,axis=0),1)
        self.TDE_matrix = numerator / denominator

    def update_buffers(self,states,actions):
        '''Store states, actions, and actor weights in buffers'''
        #print("Updated observations: "+str(states))
        #print("Updated actions: "+str(actions))
        #print("Updated actor weights: "+str(self.actor.get_weights()))
        self.state_buffer.append(states)
        self.action_buffer.append(actions)
        self.actor_weights_buffer.append(self.actor.get_weights())

    def critic_update(self,states,new_states,rewards):
        '''
        Batch update of the critic
        Arguments: consecutive states, reward, number of training epochs, batch size
        '''
        for i in range(self.n_train_epochs):
            nV = self.critic(new_states).numpy()
            TD_targets = rewards + self.gamma*nV
            loss = self.critic.fit(x=states,y=TD_targets,validation_split=0.3,batch_size=100,epochs=self.n_train_epochs,verbose=0)

    def TDE_matrix_update(self,states,new_states,rewards):
        '''
        Updates the TDE matrix with the most recent local TD errors
        and computes the delayed team average TD error
        Arguments: consecutive states, reward_scaled
        Returns: TD error matrix
        '''
        nV = self.critic(new_states).numpy()
        V = self.critic(states).numpy()
        TD_error = rewards + self.gamma*nV - V

        if self.buffer_size >= 1:
            self.TD_error_team = np.mean(self.TDE_matrix[-1,self.in_nodes],axis=0) if self.local_policy_optimization else np.mean(self.TDE_matrix[-1],axis=0)
            self.TDE_matrix[1:] = np.array(self.TDE_matrix[:-1])
            self.TDE_matrix[0] = np.zeros_like(self.TDE_matrix[0])
            self.TDE_matrix[0,self.agent_index] = TD_error
        else:
            self.TD_error_team = TD_error

        return self.TDE_matrix

    def actor_update(self):
        '''Batch update of the actor based on data stored in buffers'''

        if len(self.actor_weights_buffer) > self.buffer_size:
            'Retrieve data from buffers'
            states = self.state_buffer[0]
            actions = self.action_buffer[0]
            self.actor.set_weights(self.actor_weights_buffer[0])
            #print("Applied states: "+str(states))
            #print("Applied actions: "+str(actions))
            #print("Applied TD errors: "+str(self.TD_error_team))
            #print("Actor weights before update: "+str(self.actor.get_weights()))
            '''
            Compute delayed gradient and update current weights
            - apply that W_delayed_new = W_delayed_old + gradient_step
            - note that gradient_step = W_delayed_new - W_delayed_old
            '''
            self.actor.train_on_batch(states,actions,sample_weight=self.TD_error_team)
            #print("Actor weights after update: "+str(self.actor.get_weights()))
            new_weights = []
            for i in range(len(self.actor.trainable_weights)):
                gradient_step = self.actor.get_weights()[i]-self.actor_weights_buffer[0][i]
                #print("Gradient: "+str(gradient_step))
                new_weights.append(self.actor_weights_buffer[-1][i]+gradient_step)
            self.actor.set_weights(new_weights)
            #print("New actor weights: "+str(self.actor.get_weights()))
            'Delete data from the buffers'
            del self.actor_weights_buffer[0]
            del self.state_buffer[0]
            del self.action_buffer[0]
