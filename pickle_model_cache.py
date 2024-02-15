import pickle
from transformers import AutoTokenizer, AutoModel

# Path to the directory containing the model files
model_directory = "/path/to/your/model/directory"

# Define a function to load the model
def load_model():
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModel.from_pretrained(model_directory)
    return tokenizer, model

# Load the model and tokenizer, or load from cache if available
try:
    with open(f"{model_directory}/tokenizer_model_cache.pkl", "rb") as f:
        tokenizer, model = pickle.load(f)
except FileNotFoundError:
    tokenizer, model = load_model()
    with open(f"{model_directory}/tokenizer_model_cache.pkl", "wb") as f:
        pickle.dump((tokenizer, model), f)

# Now you can use the loaded tokenizer and model for inference
