from abc import ABC, abstractmethod
from typing import Union, Type
import numpy as np

np_array = np.ndarray[np.float64]

class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def compute(x : Union[float, np_array]) -> Union[float, np_array]:
        pass

    @staticmethod
    @abstractmethod
    def derivate(x : Union[float, np_array]) -> Union[float, np_array]:
        pass


class sigmoid(ActivationFunction):
    @staticmethod
    def compute(x : Union[float, np_array]) -> Union[float, np_array]:
        '''Overriden method.'''
        result = 1 / (1 + np.exp(-x))
        return(result)

    @staticmethod
    def derivate(x : Union[float, np_array]) -> Union[float, np_array]:
        result = x * (1 - x)
        return(result)

class relu(ActivationFunction):
    @staticmethod
    def compute(x : Union[float, np_array]) -> Union[float, np_array]:
        return(np.maximum(0, x))
    
    @staticmethod
    def derivate(x : Union[float, np_array]) -> Union[float, np_array]:
        return((x > 0).astype(np.float64))

class tanh(ActivationFunction):
    @staticmethod
    def compute(x : Union[float, np_array]) -> Union[float, np_array]:
        return(np.tanh(x))
    
    @staticmethod
    def derivate(x : Union[float, np_array]) -> Union[float, np_array]:
        return(1 - x**2)

__all__ : list[Type[ActivationFunction]] = [sigmoid, relu, tanh]