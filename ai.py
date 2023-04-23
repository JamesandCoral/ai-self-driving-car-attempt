



import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

class Lmao(nn.Module):
    
    def __init__(self, input_size, nb_action):
        super(Lmao, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)#change the number 30 (number of neruons in hidden layers) play around maybe get better results
        self.fc2 = nn.Linear(30, nb_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values
    
    
class memories(object):
    
    def __init__(self, brain_size,):
        self.brain_size = brain_size
        self.memory = []
        
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.brain_size:
            del self.memory[0]
            
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)