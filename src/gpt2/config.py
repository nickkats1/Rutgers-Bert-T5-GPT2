"""GPT2 Config"""

import torch

MODEL_NAME = "gpt2"
BATCH_SIZE = 12
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
SEED = 42
DATA_PATH = "data/guardian_headlines.csv"
OUTPUT_DIR = "src/gpt2/models/"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
