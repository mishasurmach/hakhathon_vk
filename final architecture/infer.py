from config import Config
import numpy as np
from data_preprocesser import DataPreprocessor
import joblib

def load_model(filename):
    model = joblib.load(filename)
    print(f"Model loaded from {filename}")
    return model


if __name__ == "__main__":
    preprocessor = DataPreprocessor() 
    query = input("Enter query: ")
    passage = input("Enter passage: ")

    datum = preprocessor.preprocess_single(query, passage)  
    model = load_model(Config.MODEL_TO_INFERENCE_PATH)

    y_pred = model.predict(np.array(datum).reshape(1,-1))

    if y_pred <=0.5:
        print('Negative passage.')
    else:
        print('Positive passage.')
