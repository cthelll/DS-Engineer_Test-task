from models.interface import DigitInterface
import numpy as np
import random

class RandModel(DigitInterface):
    def predict(self, img: np.ndarray) -> int:
        if img.shape != (28, 28, 1):
            raise ValueError("Image must be 28x28x1")
        return random.randint(0, 9)
