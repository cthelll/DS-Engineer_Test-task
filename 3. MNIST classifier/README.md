# MNIST Classifier Documentation

## Overview
This project implements a digit classifier for the MNIST dataset using three different algorithms:

1. **Convolutional Neural Network (CNN):** A deep learning approach for image recognition.
2. **Random Forest (RF):** A classical machine learning model.
3. **Random Predictor (RAND):** A baseline model that predicts random digits.

The classifier is designed to support the addition of new algorithms in a modular way by implementing a shared interface.

---

## Features
- **Digit Classification:** Supports classification for MNIST images (28x28 grayscale).
- **Modular Design:** New models can be added by implementing the `DigitInterface`.
- **Terminal-based Prediction:** Use the `example.py` script to test the classifier.

---

## Repository Structure

```
mnist-classifier/
├── models/
│   ├── __init__.py         # Initializes the models package
│   ├── interface.py        # Interface for classification models
│   ├── cnn.py              # CNN model implementation
│   ├── rf.py               # Random Forest model implementation
│   └── rand.py             # Random prediction model implementation
├── classifier.py           # Main DigitClassifier class
├── example.py              # Example usage of the classifier
├── README.md               # Documentation
└── requirements.txt        # Dependencies
```

---

## Requirements

### Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate   # For Linux/macOS
venv\Scripts\activate      # For Windows
```

### Install Dependencies

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### Dependencies
- `numpy`
- `tensorflow`
- `scikit-learn`
- `joblib`

---

## Usage

### 1. Prepare Pre-trained Models
- **CNN Model:** Train a Convolutional Neural Network on the MNIST dataset and save it as `models/cnn.h5`.
- **Random Forest Model:** Train a Random Forest classifier on the MNIST dataset and save it as `models/rf.joblib`.

### 2. Run the Example
Execute the `example.py` script to test the classifier:

```bash
python example.py
```

**Example Output:**
```
CNN Prediction: 8
RF Prediction: 0
Random Prediction: 9
```

---

## Adding a New Model

1. **Create a New Model File:**
   Add a new file in the `models/` directory (e.g., `new_model.py`).

2. **Implement the Interface:**
   Implement the `DigitInterface` with a `predict` method:

   ```python
   from models.interface import DigitInterface
   import numpy as np

   class NewModel(DigitInterface):
       def __init__(self, path: str):
           # Load your model here
           pass

       def predict(self, img: np.ndarray) -> int:
           # Implement prediction logic
           pass
   ```

3. **Integrate with the Classifier:**
   Update `classifier.py` to include the new model.

   ```python
   from models.new_model import NewModel

   class DigitClassifier:
       def __init__(self, algo: str):
           if algo == 'cnn':
               self.model = CNNModel('models/cnn.h5')
           elif algo == 'rf':
               self.model = RFModel('models/rf.joblib')
           elif algo == 'rand':
               self.model = RandModel()
           elif algo == 'new':
               self.model = NewModel('path/to/new/model')
           else:
               raise ValueError("Unsupported algorithm")
   ```

---

## Training the Models

### Train CNN
Use TensorFlow/Keras to train the CNN model:

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
y_train = y_train.astype('int')

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5)
model.save('models/cnn.h5')
```

### Train Random Forest
Use scikit-learn to train the Random Forest model:

```python
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.datasets import mnist
import joblib

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)
joblib.dump(rf, 'models/rf.joblib')
```

