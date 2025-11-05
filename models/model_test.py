# import os
# from tensorflow.keras.models import load_model

# def is_valid_h5(path):
#     with open(path, "rb") as f:
#         return f.read(8) == b"\x89HDF\r\n\x1a\n"

# def is_valid_keras_zip(path):
#     with open(path, "rb") as f:
#         return f.read(4) == b"\x50\x4B\x03\x04"  # ZIP signature

# def find_model():
#     model_dir = "models"
#     files = os.listdir(model_dir)
#     for fname in files:
#         fpath = os.path.join(model_dir, fname)
#         if not os.path.isfile(fpath):
#             continue
#         if fname.endswith(".keras") and is_valid_keras_zip(fpath):
#             print(f"✅ Found valid Keras zip: {fname}")
#             return fpath
#         elif fname.endswith(".h5") and is_valid_h5(fpath):
#             print(f"✅ Found valid HDF5 model: {fname}")
#             return fpath
#     raise FileNotFoundError("❌ No valid model file found in models/")

# try:
#     model_path = find_model()
#     model = load_model(model_path, compile=False)
#     print("✅ Model loaded successfully.")
#     model.summary()
# except Exception as e:
#     print(f"❌ Failed to load model: {e}")

from tensorflow.keras.models import load_model

# Load from working .h5 file
model = load_model("models/dkt_model_pretrained.h5", compile=False)

# Re-save as proper .keras format
model.save("models/dkt_model_clean.keras", save_format="keras")