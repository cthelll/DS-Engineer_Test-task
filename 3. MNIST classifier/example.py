import numpy as np
from classifier import DigitClassifier

def main():
    img = np.random.randint(0, 256, size=(28, 28, 1)).astype('float32')
    
    try:
        cnn = DigitClassifier('cnn')
        print(f'CNN Prediction: {cnn.predict(img)}')
    except Exception as e:
        print(f'CNN Prediction failed: {e}')
    
    try:
        rf = DigitClassifier('rf')
        print(f'RF Prediction: {rf.predict(img)}')
    except Exception as e:
        print(f'RF Prediction failed: {e}')
    
    try:
        rand = DigitClassifier('rand')
        print(f'Random Prediction: {rand.predict(img)}')
    except Exception as e:
        print(f'Random Prediction failed: {e}')

if __name__ == '__main__':
        main()
