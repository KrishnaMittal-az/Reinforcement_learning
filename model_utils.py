# model_utils.py
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def load_student_model(model_path="models/dkt_model_working.keras"):
    try:
        # If we have the working model, load it directly
        return load_model(model_path)
    except:
        # Fallback to building from scratch if needed
        input_layer = Input(shape=(None, 100))  # Adjust dimensions as needed
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=input_layer, outputs=output)
        model.load_weights("models/dkt_model_pretrained.h5")
        return model

def predict_student_performance(model, student_history):
    """
    Predict student performance based on their history
    student_history: List of 100-dimensional vectors representing student's interaction history
    Returns: Probability of correct answer (0-1)
    """
    # Convert to numpy array and add batch dimension
    input_data = np.array([student_history])
    return model.predict(input_data, verbose=0)[0][0]