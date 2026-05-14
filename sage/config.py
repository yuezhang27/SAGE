import os

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4o-mini"
STRONG_MODEL = "gpt-4o-mini"
TOP_K_PATHS = 5
MAX_DEBATE_ROUNDS = 3
SIGMA_TRIGGER = 1.0
SIGMA_CONVERGE = 0.5
MAX_REPAIR_ATTEMPTS = 3
AUTO_APPROVE = True
DATA_BANK_DIR = "data/data_bank"
KG_PATH = "data/mock_kg.graphml"
OUTPUT_DIR = "outputs"
