import tensorflow as tf
import numpy as np
import math
from helper import dlrelu


# Hyper Parameters
LAYER1_SIZE = 5
LAYER2_SIZE = 5
LEARNING_RATE = 1e-2
TAU = 1e-2
BATCH_SIZE = 64

class ActorNetwork:
    """docstring for ActorNetwork"""
    def __init__(self,sess,state_dim,action_dim):

        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        # create actor network
        self.state_input,self.action_output,self.net = self.create_network(state_dim,action_dim,'actor')

        # create target actor network
        self.target_state_input,self.target_action_output,self.target_update,self.target_net = self.create_target_network(state_dim,action_dim,self.net,'atarget')

        # define training rules
        self.create_training_method()

        self.sess.run(tf.global_variables_initializer())

        self.update_target()
        #self.load_network()

    def create_training_method(self):
        self.q_gradient_input = tf.placeholder("float",[None,self.action_dim])
        self.parameters_gradients = tf.gradients(self.action_output,self.net,self.q_gradient_input)
        self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,self.net))

    def create_network(self,state_dim,action_dim,scope):
        layer1_size = LAYER1_SIZE
        layer2_size = LAYER2_SIZE

        with tf.variable_scope(scope) as s:
            state_input = tf.placeholder("float",[None,state_dim])

            h1 = tf.contrib.layers.fully_connected(state_input,layer1_size,tf.nn.elu)
            h2 = tf.contrib.layers.fully_connected(h1,layer2_size,tf.nn.elu)
            action_output = tf.contrib.layers.fully_connected(h2,action_dim,tf.tanh,weights_initializer=tf.random_uniform_initializer(-3e-3,3e3))
            net = [v for v in tf.trainable_variables() if scope in v.name]

        return state_input,action_output,net

    def create_target_network(self,state_dim,action_dim,net,scope):
        state_input,action_output,target_net = self.create_network(state_dim,action_dim,scope)
        # updating target netowrk
        target_update = []
        for n1,n2 in zip(net,target_net):
            # theta' <-- tau*theta + (1-tau)*theta'
            target_update.append(n2.assign(tf.add(tf.multiply(TAU,n1),tf.multiply((1-TAU),n2))))

        return state_input,action_output,target_update,target_net

    def update_target(self):
        self.sess.run(self.target_update)

    def train(self,q_gradient_batch,state_batch):
        self.sess.run(self.optimizer,feed_dict={
            self.q_gradient_input:q_gradient_batch,
            self.state_input:state_batch
            })

    def actions(self,state_batch):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:state_batch
            })

    def action(self,state):
        return self.sess.run(self.action_output,feed_dict={
            self.state_input:[state]
            })[0]


    def target_actions(self,state_batch):
        return self.sess.run(self.target_action_output,feed_dict={
            self.target_state_input:state_batch
            })

    # f fan-in size
    def variable(self,shape,f):
        return tf.Variable(tf.random_uniform(shape,-1/math.sqrt(f),1/math.sqrt(f)))
'''
    def load_network(self):
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state("saved_actor_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print "Successfully loaded:", checkpoint.model_checkpoint_path
        else:
            print "Could not find old network weights"
    def save_network(self,time_step):
        print 'save actor-network...',time_step
        self.saver.save(self.sess, 'saved_actor_networks/' + 'actor-network', global_step = time_step)

'''


