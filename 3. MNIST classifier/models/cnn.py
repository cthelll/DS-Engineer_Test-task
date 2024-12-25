from models.interface import DigitInterface
import numpy as np
from tensorflow.keras.models import load_model

class CNNModel(DigitInterface):
    def __init__(self, path: str):
        self.model = load_model(path)

    def predict(self, img: np.ndarray) -> int:
        if img.shape != (28, 28, 1):
            raise ValueError("Image must be 28x28x1")
        img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
        pred = self.model.predict(img)
        return int(np.argmax(pred, axis=1)[0])
