import numpy as np
from dense_layer import DensePhaseThree
from input_layer import Input
import time


class Model:
    
    def __init__(self):
        self.layers = []
        self.count = 1
        
        
               
    def add(self, layer:DensePhaseThree):
        layer.previous_layer = None
        '''
        set last item in list as previous layer
        '''
        self.layers.append(layer)

    def show_layers(self):
        print("=================== LAYER INFO ===================")
        for index, layer in enumerate(self.layers):
            print(f"""
                  Index::{index}
                        Layer Name:: {layer.layer_name} {type(layer.layer_name)}
                        Layer Address:: {layer}  {type(layer)}      
                        Layer units::{layer.units} {type(layer.units)}
                        Previous Layer::{layer.previous_layer} {type(layer.previous_layer)}
                        
                        """)
                
            
    def compile_(self, input_data:np.array): 
        '''
        connecting all the layers
        
        running forward pass for all layers
        
        and connect input layer with input data
        '''       
        
        #  connecting all the layers
        
        for i in range(len(self.layers)-1):
            self.layers[i+1].previous_layer = self.layers[i]

        # DONE WITH CONNECTING ALL THE LAYERS
        
        
        if isinstance(self.layers[0], Input) and self.layers[0].units == input_data.shape[0]:
            self.layers[0].input_data = input_data
                
        for layer in self.layers:
            layer.forward_pass() 
            
            
    def loss_calc(self, y_true:np.array, y_pred:np.array):
        '''
        calculating mean squared error
        '''
        return np.mean(np.square(y_true - y_pred))   
            
    def fit(self):
        '''
        initialize weights and biases with 0 & 1,
        take weights and add 100 to it, 
        do it 5 times(epochs) and
        dont update  weights
        # '''
        for layer in reversed(self.layers):
            if isinstance(layer, Input):
                print("no weights and biases update for input layer\n")
                continue
            
            print("weights before updating = ", layer._DensePhaseThree__weights, "\n")
            layer._DensePhaseThree__weights = layer._DensePhaseThree__weights * 100
            print("weights after updating = ", layer._DensePhaseThree__weights, "\n")
            time.sleep(1)
            
            print("biases before updating = ", layer._DensePhaseThree__biases, "\n")
            layer._DensePhaseThree__biases = layer._DensePhaseThree__biases + 200
            print("biases after updating = ", layer._DensePhaseThree__biases, "\n")
            print("\n","="*50,"\n")
            
            
