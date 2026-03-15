"""GPT2 Config"""


MODEL_NAME = "gpt-2"
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 5e-5
MAX_LENGTH = 128
SEED = 42
DATA_PATH = "data/guardian_headlines.csv"
OUTPUT_DIR = "src/gpt2/models/"
DEVICE = "cuda:0"



