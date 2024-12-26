from .activation_functions import ActivationFunction, relu
from .activation_functions import __all__ as functions
from typing import TypeVar, Generic, Type, Self
import numpy as np
import json
import os


function_t = TypeVar('function_t', bound=ActivationFunction)

class Layer(Generic[function_t]):
    _DEFAULT_ACTIVATION_FUNCTION = relu
    _FILENAME_START = "Layer"

    @staticmethod
    def default_activation_function() -> Type[ActivationFunction]:
        return(Layer._DEFAULT_ACTIVATION_FUNCTION)

    @classmethod
    def load(cls, dirname : str, index : int) -> Self:
        filename = Layer._FILENAME_START + str(index)

        weights = np.load(os.path.join(dirname, filename + ".npy"))

        with open(os.path.join(dirname, filename + ".json"), 'r') as file:
            data = json.load(file)
        
        n_neurons = data["Número de neuronas"]
        n_inputs = data["Número de entradas"]
        function_reference = data["Función de activación"]
        function = cls._get_activation_function(function_reference)

        layer = Layer(n_neurons, n_inputs, function)
        layer._weights = weights

        return(layer)

    @classmethod
    def _get_activation_function(cls, function_reference : str) -> Type[ActivationFunction]:
        for function in functions:
            if (function.__name__ == function_reference):
                return(function)

    def __init__(self, n_neurons : int, n_inputs : int, function_type : Type[ActivationFunction] = _DEFAULT_ACTIVATION_FUNCTION) -> None:
        self._n_neurons = n_neurons
        self._n_inputs = n_inputs
        self._function = function_type
        self._generate_random()

    def save(self, dirname : str, index : int) -> None:
        os.makedirs(dirname, exist_ok=True)
        filename = Layer._FILENAME_START + str(index)
        with open(os.path.join(dirname, filename + ".npy"), 'wb') as file:
            np.save(file, self.weights)

        with open(os.path.join(dirname, filename + ".json"), 'w') as file:
            json.dump(self.to_dict(), file)

    def to_dict(self) -> dict:
        data = {
            "Número de neuronas": self.n_neurons,
            "Número de entradas": self.n_inputs,
            "Función de activación": self.function.__name__
        }
        return(data)

    @property
    def n_neurons(self) -> int:
        return(self._n_neurons)

    @property
    def n_inputs(self) -> int:
        return(self._n_inputs)
    
    @property
    def weights(self) -> np.ndarray[np.float64]:
        return(self._weights)
    
    @property
    def function(self) -> ActivationFunction:
        return(self._function)
    
    @weights.setter
    def weights(self, value : np.ndarray[np.float64]) -> None:
        self._weights = value

    def __str__(self) -> str:
        s = "Función: " + self._function.__name__ + "\n"
        s += str(self.weights)
        return s

    def _generate_random(self) -> None:
        random_weights = 2 * np.random.random((self._n_inputs, self._n_neurons)) - 1
        self._weights = random_weights

    def compute(self, input : np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        result = self._function.compute(np.dot(input, self._weights))
        return(result)
    
    def adjust(self, quantity : np.ndarray[np.float64]) -> None:
        self._weights += quantity