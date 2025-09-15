# Suppress transformers warnings about missing PyTorch/TensorFlow
import os

# Set transformers verbosity to error level to suppress warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
