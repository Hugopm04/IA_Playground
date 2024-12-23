from abc import ABC, abstractmethod
import math

class ActivationFunction(ABC):
    @staticmethod
    @abstractmethod
    def __call__(x : float) -> float:
        pass

    @staticmethod
    @abstractmethod
    def derivate(x : float) -> float:
        pass


class sigmoid(ActivationFunction):
    @staticmethod
    def __call__(x : float) -> float:
        '''Overriden method.'''
        result = 1 / (1 + math.exp(-x))
        return(result)

class linear(ActivationFunction):
    @staticmethod
    def __call__(x : float) -> float:
        return(x)