import numpy as np
import tensorflow as tf
# Helper Function------------------------------------------------------------------------------------------------------------
# Copies one set of variables to another.
# Used to set worker network parameters to those of global network.

def dlrelu(x, alpha=0.1):
  return tf.nn.relu(x) - alpha * tf.nn.relu(0.05-x) - (1 - alpha) *  tf.nn.relu(x-0.95) 

class RunningStats:

    def __init__(self):
        self.n = 0
        self.old_m = 0
        self.new_m = 0
        self.old_s = 0
        self.new_s = 0

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = 0
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 0.0

    def standard_deviation(self):
        return np.sqrt(self.variance())
        
    def normalize(self,x):
        self.push(x)
        return (x - self.mean()) / (self.standard_deviation()+1e-3) if self.n > 1 else x

        
def n_step_transition(episode_buffer,n_step,gamma):
    _,_,_,s1,done = episode_buffer[-1]
    s,action,_,_,_ = episode_buffer[-1-n_step]
    r = 0
    for i in range(n_step):
      r += episode_buffer[-1-n_step+i][2]*gamma**i
    return [s,action,r,s1,done]


