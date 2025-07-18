import joblib
import numpy as np

def load_model(path="models/best_model.pkl"):
    return joblib.load(path)

def predict(model, user_input):
    return model.predict(np.array([user_input]))
