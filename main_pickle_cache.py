import pickle

# Path to the directory containing the model files
model_directory = "/path/to/your/model/directory"

# Load the tokenizer and model from the cache
try:
    with open(f"{model_directory}/tokenizer_model_cache.pkl", "rb") as f:
        tokenizer, model = pickle.load(f)
except FileNotFoundError:
    print("Cached tokenizer and model not found. Please ensure that they have been cached.")
    tokenizer, model = None, None

# Now you can use the loaded tokenizer and model for inference
