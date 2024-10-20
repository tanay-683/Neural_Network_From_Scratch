import numpy as np

class Input:


    '''
    making it for only one dimensional input
    '''
    def __init__(self, input_shape:int, layer_name:str):
        
        self.layer_name = layer_name
        self.units = input_shape
        self.input_data:np.array = np.zeros(self.units) # None only at time of initialization
        
        # these parameters doesnt exist for input layer
        self.previous_layer = None
        self.previous_layer_units = None
        self.__weights = None
        self.__biases = None
        
        
    def forward_pass(self):
        return self.input_data