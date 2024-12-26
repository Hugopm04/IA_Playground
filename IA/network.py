from abc import ABC, abstractmethod
import numpy as np
from typing import Self

class Network(ABC):

    @classmethod
    @abstractmethod
    def load(cls, dirname : str) -> Self:
        pass

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @abstractmethod
    def save(self, dirname : str) -> None:
        pass

    @abstractmethod
    def think(self, inputs : np.ndarray[np.float64]) -> np.ndarray[np.float64]:
        pass

    @abstractmethod
    def database_train(self, known_inputs : np.ndarray[np.float64], known_outputs : np.ndarray[np.float64], iterations : int) -> None:
        pass

    @abstractmethod
    def real_time_train(self) -> None:
        pass