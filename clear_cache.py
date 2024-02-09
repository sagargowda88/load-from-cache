from joblib import Memory

# Path to the directory containing the model files
model_directory = "/path/to/your/model/directory"

# Create a memory cache
memory = Memory(location=model_directory, verbose=0)

# Clear the cache
memory.clear()
