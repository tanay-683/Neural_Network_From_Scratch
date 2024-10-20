import numpy as np
from input_layer import Input


class DensePhaseThree:
    
    def __init__(self, units:int, activation:str, layer_name:str):
        self.units = units
        self.activation = activation
        self.layer_name = layer_name
        
        self.previous_layer:DensePhaseThree = None
        self.previous_layer_values = None
        self.previous_layer_units = None
        self.__weights = None
        self.__biases = None
        
        
    
    def forward_pass(self) -> np.array:

        #  determining previous layer units
        
        self.previous_layer_units = self.previous_layer.units
        
        # if the previous layer is input layer then previous layer values will be input data
        if isinstance(self.previous_layer, Input):
            self.previous_layer_values = self.previous_layer.input_data

        # if the previous layer is dense layer then previous layer values will be activation output of previous layer
        if isinstance(self.previous_layer, DensePhaseThree):
            self.previous_layer_values = self.previous_layer.activation_output

                
        # initializing weights and biases of this layer
        self.__weights, self.__biases = self.__initialize_weights_and_biases(self.previous_layer_units)
        
        
        # calculating output ---> WiXi + Bo 
        self.output = np.add(np.dot(self.__weights, self.previous_layer_values), self.__biases)


        # applying activation function on output
        if self.activation.lower() == 'relu':
            activation_func = np.vectorize(self.__relu_activation)
            self.activation_output = activation_func(self.output).astype(np.float32)
            
        elif self.activation.lower() == 'sigmoid':
            activation_func = np.vectorize(self.__sigmoid_activation)
            self.activation_output = activation_func(self.output).astype(np.float32)
            
        else: # linear
            self.activation_output = self.output
            
        print("============ Layer Name: ", self.layer_name.upper(), " ============")
        print("\n\ncurrent_layer_units = ", self.units, "\n")
        print("biases_shape = ", self.__biases.shape, "\n")
        print("weights_shape = ", self.__weights.shape, "\n")
        print("previous layer units = ", self.previous_layer_units, "\n")
        print("previous layer values = ", self.previous_layer_values, "\n")
        print("Output for next layer is =", self.activation_output, "\n")
        print("\n","="*50,"\n")
        

        
        return self.activation_output

        
    def __initialize_weights_and_biases(self,previous_layer_units:int):
        
        '''
        Note: using `he normal` initialization
        '''
        
        biases = np.random.randn(self.units)*np.sqrt(2/previous_layer_units)
        weights = np.random.randn(self.units, previous_layer_units)*np.sqrt(2/previous_layer_units)
        
        return weights, biases
        

    def __relu_activation(self, layer_output:np.array):
        return max(0, layer_output)
    
    
    def __sigmoid_activation(self, layer_output:np.array):
        if layer_output>0:
            return 1/(1+np.exp(-layer_output)) # 1/(1+e^-z)
        else:
            return np.exp(layer_output)/(1+np.exp(layer_output)) # e^z / (1+e^z)
    