from .activation_functions import ActivationFunction, relu
from typing import TypeVar, Generic, Type
import numpy as np


function_t = TypeVar('function_t', bound=ActivationFunction)

class Layer(Generic[function_t]):
    _DEFAULT_ACTIVATION_FUNCTION = relu

    @staticmethod
    def default_activation_function() -> Type[ActivationFunction]:
        return(Layer._DEFAULT_ACTIVATION_FUNCTION)

    def __init__(self, n_neurons : int, n_inputs : int, function_type : Type[ActivationFunction] = _DEFAULT_ACTIVATION_FUNCTION) -> None:
        self._n_neurons = n_neurons
        self._n_inputs = n_inputs
        self._function = function_type
        self._generate_random()

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
    
    def __str__(self) -> str:
        return(str(self._weights))

    def _generate_random(self) -> None:
        random_weights = 2 * np.random.random((self._n_inputs, self._n_neurons)) - 1
        self._weights = random_weights

    def compute(self, input : np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        result = self._function.compute(np.dot(input, self._weights))
        return(result)
    
    def adjust(self, quantity : np.ndarray[np.float64]) -> None:
        self._weights += quantity