import os
import requests
from tqdm import tqdm

def download_file(url, filename):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()
    
    if total_size != 0 and progress_bar.n != total_size:
        print("ERROR: Failed to download the file completely")
        return False
    
    print(f"✓ Successfully downloaded {filename}")
    return True

# URL of the model file (you may need to replace this with the actual URL)
MODEL_URL = "https://huggingface.co/your-username/your-model/resolve/main/dkt_model_pretrained.h5"
MODEL_PATH = "./models/dkt_model_pretrained.h5"

if __name__ == "__main__":
    print("Downloading pre-trained model...")
    if download_file(MODEL_URL, MODEL_PATH):
        print("\nModel downloaded successfully!")
        print(f"Model saved to: {os.path.abspath(MODEL_PATH)}")
    else:
        print("\nFailed to download the model. Please check your internet connection and try again.")
        print("You may need to manually download the model file and place it in the 'models' folder.")
