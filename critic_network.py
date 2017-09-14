import tensorflow as tf
import numpy as np
import math
from helper import dlrelu


LAYER1_SIZE = 20
LAYER2_SIZE = 10
LEARNING_RATE = 5e-3
TAU = 1e-2
L2 = 0.01

class CriticNetwork:
    """docstring for CriticNetwork"""
    def __init__(self,sess,state_dim,action_dim,atoms,z):
        self.time_step = 0
        self.sess = sess
        self.atoms = atoms
        self.z = z
        # create q network
        self.state_input,\
        self.action_input,\
        self.q_value_output,\
        self.net = self.create_q_network(state_dim,action_dim,atoms,'critic')

        # create target q network (the same structure with q network)
        self.target_state_input,\
        self.target_action_input,\
        self.target_q_value_output,\
        self.target_update = self.create_target_q_network(state_dim,action_dim,atoms,self.net,'ctarget')

        self.create_training_method()

        # initialization
        self.sess.run(tf.global_variables_initializer())

        self.update_target()

    def create_training_method(self):
        # Define training optimizer
        self.m_input = tf.placeholder("float",[None,self.atoms])
        weight_decay = tf.add_n([L2 * tf.nn.l2_loss(var) for var in self.net])
        self.cost = -tf.reduce_sum(self.m_input * tf.log(self.q_value_output)) + weight_decay
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)
        self.action_gradients = tf.gradients(tf.reduce_sum(self.z * self.q_value_output,axis=1),self.action_input)

    def create_q_network(self,state_dim,action_dim,atoms,scope):
        # the layer size could be changed
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE
        with tf.variable_scope(scope) as s:
            state_input = tf.placeholder("float",[None,state_dim])
            action_input = tf.placeholder("float",[None,action_dim])

            h1 = tf.contrib.layers.fully_connected(state_input,layer1_size,tf.nn.elu)
            h2 = tf.contrib.layers.fully_connected(tf.concat([h1,action_input],axis=1),layer2_size,tf.nn.elu)
            q_value_output = tf.contrib.layers.fully_connected(h2,atoms,tf.nn.softmax,weights_initializer=tf.random_uniform_initializer(-3e-3,3e-3))
            net = [v for v in tf.trainable_variables() if scope in v.name]
        
        return state_input,action_input,q_value_output,net

    def create_target_q_network(self,state_dim,action_dim,atoms,net,scope):
        state_input,action_input,q_value_output,target_net = self.create_q_network(state_dim,action_dim,atoms,scope)

        target_update = []
        for n1,n2 in zip(net,target_net):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(n2.assign(tf.add(tf.multiply(TAU,n1),tf.multiply((1-TAU),n2))))

        return state_input,action_input,q_value_output,target_update

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,m_batch,state_batch,action_batch):
        self.time_step += 1
        self.sess.run(self.optimizer,feed_dict={
            self.m_input:m_batch,
            self.state_input:state_batch,
            self.action_input:action_batch
            })

    def gradients(self,state_batch,action_batch):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch
            })[0]

    def target_q(self,state_batch,action_batch):
        return self.sess.run(self.target_q_value_output,feed_dict={
            self.target_state_input:state_batch,
            self.target_action_input:action_batch
            })

    def q_value(self,state_batch,action_batch):
        return self.sess.run(self.q_value_output,feed_dict={
            self.state_input:state_batch,
            self.action_input:action_batch})

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_critic_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"

    def save_network(self,time_step):
        print 'save critic-network...',time_step
        self.saver.save(self.sess, 'saved_critic_networks/' + 'critic-network', global_step = time_step)
'''
