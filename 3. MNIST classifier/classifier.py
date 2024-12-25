from models.cnn import CNNModel
from models.rf import RFModel
from models.rand import RandModel
import numpy as np

class DigitClassifier:
    def __init__(self, algo: str):
        if algo == 'cnn':
            self.model = CNNModel('models/cnn.h5')
        elif algo == 'rf':
            self.model = RFModel('models/rf.joblib')
        elif algo == 'rand':
            self.model = RandModel()
        else:
            raise ValueError("Choose 'cnn', 'rf', or 'rand'")

    def predict(self, img: np.ndarray) -> int:
        return self.model.predict(img)
