from model import Model
from input_layer import Input
from dense_layer import DensePhaseThree
import numpy as np 

input_ = np.random.rand(5)


model = Model()

model.add(Input(input_shape=input_.shape[0], layer_name="input_"))
model.add(DensePhaseThree(units = 5,activation = "relu", layer_name="first"))
model.add(DensePhaseThree(units = 3,activation = "relu", layer_name="second"))
model.add(DensePhaseThree(units = 1,activation = "SIGMOID", layer_name="third"))


model.compile_(input_)
model.fit()
