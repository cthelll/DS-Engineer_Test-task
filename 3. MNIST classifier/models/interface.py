from abc import ABC, abstractmethod
import numpy as np

class DigitInterface(ABC):
    @abstractmethod
    def predict(self, img: np.ndarray) -> int:
        pass
