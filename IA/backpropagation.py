from .activation_functions import ActivationFunction
from .neuroal_layer import Layer
from .network import Network
from typing import Type, Self
import numpy as np
import json
import os


class BackPropagationNetwork(Network):
    _FILENAME = "network.json"

    @classmethod
    def load(cls, dirname : str) -> Self:
        with open(os.path.join(dirname, BackPropagationNetwork._FILENAME), 'r') as file:
            data = json.load(file)
        
        network = BackPropagationNetwork()
        for i in range(data["Número de capas"]):
            network._layers.append(Layer.load(dirname, i))
        
        return(network)
    
    def __init__(self):
        self._layers : list[Layer] = []

    def save(self, dirname : str) -> None:
        os.makedirs(dirname, exist_ok=True)

        for i, layer in enumerate(self._layers):
            layer.save(dirname, i)
        
        with open(os.path.join(dirname, BackPropagationNetwork._FILENAME), 'w') as file:
            json.dump(self.to_dict(), file)

    def to_dict(self) -> dict:
        data = {
            "Número de capas": self.size()
        }
        return data

    def add_layer(self, n_neurons : int, n_inputs : int, activation_function : Type[ActivationFunction] = Layer.default_activation_function()) -> None:
        self.insert_layer(self.size(), n_neurons, n_inputs, activation_function)

    def remove_layer(self, index : int) -> None:
        if (index < len(self._layers) and index >= 0):
            self._layers.pop(index)
    
    def insert_layer(self, index : int, n_neurons : int, n_inputs : int, activation_function : Type[ActivationFunction] = Layer.default_activation_function()) -> None:
        if (index <= len(self._layers) and index >= 0):
            new_layer = Layer(n_neurons, n_inputs, activation_function)
            self._layers.insert(index, new_layer)
    
    def size(self) -> int:
        return(len(self._layers))

    def real_time_train(self):
        pass

    def think(self, inputs : np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        for layer in self._layers:
            output = layer.compute(inputs)
            inputs = output
        return(output)
    
    def _thinking_proccess(self, inputs : np.ndarray[np.float64]) -> list[np.ndarray[np.float64]]:
        outputs = []
        for layer in self._layers:
            output = layer.compute(inputs)
            inputs = output
            outputs.append(output)
        return(outputs)

    def database_train(self, known_inputs : np.ndarray[np.float64], known_outputs : np.ndarray[np.float64], iterations : int):
        for iteration in range(iterations):
            outputs = self._thinking_proccess(known_inputs)

            '''As backpropagation starts with the final output we reverse both lists to iterate over them in reversed order.'''
            for i in reversed(range(self.size())):
                if i == self.size() - 1: #Output layer
                    layer_error = known_outputs - outputs[i]

                else:
                    layer_error = layer_delta.dot(self._layers[i+1].weights.T)
                
                derivate = self._layers[i].function.derivate(outputs[i])
                layer_delta = layer_error * derivate

                if i == 0: #Input layer
                    layer_adjustement = known_inputs.T.dot(layer_delta)
                else:
                    layer_adjustement = outputs[i-1].T.dot(layer_delta)
                
                self._layers[i].adjust(layer_adjustement)
    
    def __str__(self) -> str:
        START_OF_STRING = "Red neuronal: \n"
        START_OF_LINE = "\t- Pesos de la neurona "

        s = START_OF_STRING

        for i,layer in enumerate(self._layers):
            s += START_OF_LINE
            s += str(i + 1)
            s += ": " + str(layer) + "\n"
        
        return(s)
