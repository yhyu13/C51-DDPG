Standard implementation of the DDPG algorithm 
https://arxiv.org/abs/1509.02971 
Coupled with DeepMind's newest ditributional bellman equation update, chekout critic_network.py (loss function) and ddpg.py (train function) for details.
https://arxiv.org/pdf/1707.06887.pdf
FILES:
actor_network.py: The code for structure of the actor network  
critic_network.py: The code for structure of the critic network  
ou_noise.py:The random noise generator    
ddpg.py:The code for the ddpg algorithm    
gym_ddpg.py:Running ddpg on the environment    
