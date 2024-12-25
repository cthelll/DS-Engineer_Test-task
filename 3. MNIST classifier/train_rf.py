# train_rf.py
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255.0
x_val = x_val.reshape(-1, 28*28).astype('float32') / 255.0

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

joblib.dump(rf, 'models/rf.joblib')

