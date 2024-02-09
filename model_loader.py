from joblib import Memory
from transformers import AutoTokenizer, AutoModel

# Path to the directory containing the model files
model_directory = "/path/to/your/model/directory"

# Create a memory cache
memory = Memory(location=model_directory, verbose=0)

# Define a function to load the model
@memory.cache
def load_model():
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModel.from_pretrained(model_directory)
    return tokenizer, model
