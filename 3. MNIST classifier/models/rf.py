from models.interface import DigitInterface
import numpy as np
import joblib

class RFModel(DigitInterface):
    def __init__(self, path: str):
        self.model = joblib.load(path)

    def predict(self, img: np.ndarray) -> int:
        if img.shape != (28, 28, 1):
            raise ValueError("Image must be 28x28x1")
        flat = img.flatten().reshape(1, -1)
        pred = self.model.predict(flat)
        return int(pred[0])
