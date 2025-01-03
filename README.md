﻿# DS_Engineer_Test_task

This repository contains three tasks implemented for a test assignment:

1. **Counting Islands**
2. **Regression on Tabular Data**
3. **MNIST Classifier**
---

## Repository Structure

```
test-assignment/
├── counting-islands/
│   ├── island_counter.py
│   └── README.md
├── regression-tabular-data/
│   ├── data/
│   │   ├── train.csv
│   │   ├── hidden_test.csv
│   │   └── sample_submission.csv
│   ├── notebooks/
│   │   └── EDA.ipynb
│   ├── scripts/
│   │   ├── train.py
│   │   └── predict.py
│   ├── predictions/
│   │   └── predictions.csv
│   ├── README.md
│   └── requirements.txt
├── mnist-classifier/
│   ├── models/
│   │   ├── cnn.py
│   │   ├── rf.py
│   │   ├── rand.py
│   │   └── interface.py
│   ├── classifier.py
│   ├── example.py
│   ├── README.md
│   └── requirements.txt
└── README.md (this file)
```

---

## 1. Counting Islands

This script counts the number of islands in a 2D grid. An island is defined as a group of `1`s connected horizontally or vertically.

### How to Run

1. Save the script as `island_counter.py`.
2. Run the script using Python 3:
   ```bash
   python island_counter.py
   ```

3. Input test cases or modify the test cases in the code for validation.

### Example Input/Output

Input:
```
3 3
0 1 0
0 0 0
0 1 1
```

Output:
```
Number of islands: 2
```

---

## 2. Regression on Tabular Data

This project involves building a regression model to predict a target variable based on 53 anonymized features. The main objective is to minimize the RMSE on the test dataset.

### Setup Instructions

1. **Create a Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # For Linux/macOS
   venv\Scripts\activate      # For Windows
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook for EDA**:
   Open the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

4. **Train the Model**:
   Run the training script:
   ```bash
   python scripts/train.py --train_path data/train.csv --model_path models/random_forest.joblib
   ```

5. **Generate Predictions**:
   Run the prediction script:
   ```bash
   python scripts/predict.py --test_path data/hidden_test.csv --model_path models/random_forest.joblib --output_path predictions/predictions.csv
   ```

### Requirements

Dependencies are listed in the `requirements.txt` file:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- jupyter

---

## 3. MNIST Classifier

This project implements a digit classifier for the MNIST dataset using three algorithms:

1. **CNN**: A deep learning model.
2. **Random Forest (RF)**: A classical machine learning model.
3. **Random Predictor (RAND)**: A baseline model.

### Usage

1. **Prepare Pre-trained Models**:
   - Train and save a CNN model as `models/cnn.h5`.
   - Train and save a Random Forest model as `models/rf.joblib`.

2. **Run Example**:
   Execute the `example.py` script to test the classifier:
   ```bash
   python example.py
   ```

   **Example Output**:
   ```
   CNN Prediction: 8
   RF Prediction: 0
   Random Prediction: 9
   ```

### Training Instructions

#### Train CNN
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

#### Train Random Forest
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

