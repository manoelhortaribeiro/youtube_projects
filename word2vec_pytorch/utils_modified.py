import numpy as np
import sys

def q():
    sys.exit()

# define a function to count the total number of trainable parameters

'''
Count the total number of trainable parameters of the model
PARAMETER:
    - model: the model on which we count the parameter
'''
def count_parameters(model): 
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_parameters/1e6 # in terms of millions
