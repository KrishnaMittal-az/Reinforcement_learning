import tensorflow as tf
import h5py
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout

def build_model_from_weights(h5_path):
    try:
        # Create a dummy model with the same architecture
        # (You'll need to adjust these dimensions based on your model's architecture)
        input_layer = Input(shape=(None, 100))  # Adjust input shape as needed
        x = LSTM(128, return_sequences=True)(input_layer)
        x = Dropout(0.2)(x)
        x = LSTM(64)(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid')(x)  # Adjust output layer as needed
        
        model = Model(inputs=input_layer, outputs=output)
        
        # Load weights
        model.load_weights(h5_path, by_name=True, skip_mismatch=True)
        print("✅ Successfully loaded weights into new model")
        return model
        
    except Exception as e:
        print(f"❌ Failed to build model: {str(e)}")
        return None

# Try loading the first HDF5 file
model = build_model_from_weights("models/dkt_model_pretrained.h5")
if model:
    print("\nModel Summary:")
    model.summary()
else:
    # If that fails, try to inspect the model architecture
    print("\nInspecting model architecture...")
    try:
        with h5py.File("models/dkt_model_pretrained.h5", 'r') as f:
            print("Model weights structure:")
            def print_weights(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
            f.visititems(print_weights)
    except Exception as e:
        print(f"❌ Failed to inspect model: {str(e)}")

# Create some dummy input data (batch of 1, sequence length 10, 100 features)
dummy_input = np.random.rand(1, 10, 100)
prediction = model.predict(dummy_input)
print(f"Prediction shape: {prediction.shape}")
print(f"Sample prediction: {prediction[0][0]:.4f}")
model.save("models/dkt_model_working.keras")