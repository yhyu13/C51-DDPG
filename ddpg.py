# -----------------------------------
# Deep Deterministic Policy Gradient
# Author: Kaizhao Liang
# Date: 08.11.2017
# -----------------------------------
import tensorflow as tf
import numpy as np
from ou_noise import OUNoise
from critic_network import CriticNetwork
from actor_network import ActorNetwork
from replay_buffer import ReplayBuffer

from helper import *

# Hyper Parameters:

REPLAY_BUFFER_SIZE = 100000
REPLAY_START_SIZE = 50000
BATCH_SIZE = 64
GAMMA = 0.99
N_STEP = 1
ATOMS = 51
VMAX=100 # change VMAX,VMIN by each environment
VMIN=-5 

def make_session(num_cpu):
    """Returns a session that will use <num_cpu> CPU's only"""
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=num_cpu,
        intra_op_parallelism_threads=num_cpu,
        device_count = {'GPU': 0}) # no gpu required
    return tf.InteractiveSession(config=tf_config)

class DDPG:
    """docstring for DDPG"""
    def __init__(self, env):
        self.name = 'C51_DDPG' # name for uploading results
        self.env = env
        # Randomly initialize actor network and critic network
        # with both their target networks
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.atoms = ATOMS
        self.start_train = False
        
        self.v_max = VMAX
        self.v_min = VMIN
        self.delta_z = (self.v_max - self.v_min) / (self.atoms - 1.)
        self.z = np.tile(np.asarray([self.v_min + i * self.delta_z for i in range(self.atoms)]).astype(np.float32),(BATCH_SIZE,1)) # shape (BATCH_SIZE,atoms)

        self.sess = make_session(1)

        self.actor_network = ActorNetwork(self.sess,self.state_dim,self.action_dim)
        self.critic_network = CriticNetwork(self.sess,self.state_dim,self.action_dim,self.atoms,self.z)

        # initialize replay buffer
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

        # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
        self.exploration_noise = OUNoise(self.action_dim)

        self.saver = tf.train.Saver()

    def train(self):
        #print "train step",self.time_step
        # Sample a random minibatch of N transitions from replay buffer
        minibatch = self.replay_buffer.get_batch(BATCH_SIZE)
        state_batch = np.asarray([data[0] for data in minibatch])
        action_batch = np.asarray([data[1] for data in minibatch])
        reward_batch = np.asarray([data[2] for data in minibatch])
        next_state_batch = np.asarray([data[3] for data in minibatch])
        done_batch = np.asarray([data[4] for data in minibatch])

        # for action_dim = 1
        action_batch = np.resize(action_batch,[BATCH_SIZE,self.action_dim])

        # Calculate y_batch

        next_action_batch = self.actor_network.target_actions(next_state_batch)
        q_value_batch = self.critic_network.target_q(next_state_batch,next_action_batch)
        done_batch = np.asarray([0. if done else 1. for done in done_batch])
        
        Tz = np.minimum(self.v_max, np.maximum(self.v_min,reward_batch[:,np.newaxis] + GAMMA**N_STEP * self.z * done_batch[:,np.newaxis]))
        b = (Tz - self.v_min) / self.delta_z
        l,u = np.floor(b+1e-3).astype(int),np.ceil(b-1e-3).astype(int)
        #print(l)
        #print(u)
        p = q_value_batch
        m_batch = np.zeros((BATCH_SIZE,self.atoms))
        A = p * (u - b)
        B = p * (b - l)
        for i in range(BATCH_SIZE):
            for j in range(self.atoms):
                m_batch[i,l[i,j]] += A[i,j]
                m_batch[i,u[i,j]] += B[i,j]
        # Update critic by minimizing the loss L
        self.critic_network.train(m_batch.astype(np.float32),state_batch,action_batch)

        # Update the actor policy using the sampled gradient:
        action_batch_for_gradients = self.actor_network.actions(state_batch)
        q_gradient_batch = self.critic_network.gradients(state_batch,action_batch_for_gradients)
        
        q_gradient_batch *= -1.
        '''
        #print(q_gradient_batch)
        # invert gradient formula : dq = (a_max-a) / (a_max - a_min) if dq>0, else dq = (a - a_min) / (a_max - a_min)
        for i in range(BATCH_SIZE): # In our case a_max = 1, a_min = 0
            for j in range(self.action_dim):
                dq = q_gradient_batch[i,j]
                a = action_batch_for_gradients[i,j]
                if dq > 0.:
                    q_gradient_batch[i,j] *= (1-a)
                else:
                    q_gradient_batch[i,j] *= a-0
        '''
        self.actor_network.train(q_gradient_batch,state_batch)

        # Update the target networks
        self.actor_network.update_target()
        self.critic_network.update_target()

    def save_model(self, path, episode):
        #if self.episode % 10 == 1:
        self.saver.save(self.sess, path + "modle.ckpt", episode)


    def noise_action(self,state):
        # Select action a_t according to the current policy and exploration noise
        action = self.actor_network.action(state)
        return action+self.exploration_noise.noise()

    def action(self,state):
        action = self.actor_network.action(state)
        return action

    def perceive(self,state,action,reward,next_state,done):
        # Store transition (s_t,a_t,r_t,s_{t+1}) in replay buffer
        self.replay_buffer.add(state,action,reward,next_state,done)

        # Store transitions to replay start size then start training
        if self.replay_buffer.count() >  REPLAY_START_SIZE:
            self.start_train = True
            self.train()

        #if self.time_step % 10000 == 0:
            #self.actor_network.save_network(self.time_step)
            #self.critic_network.save_network(self.time_step)

        # Re-iniitialize the random process when an episode ends
        if done:
            self.exploration_noise.reset()












