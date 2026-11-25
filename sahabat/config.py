"""Central settings for Sahabat Code. Every script reads from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "projects.csv"
QUERIES_PATH = ROOT / "data" / "search_queries.csv"
ARTIFACTS = ROOT / "artifacts"
CLASSIFIER_DIR = ARTIFACTS / "classifier"
BASELINE_PATH = ARTIFACTS / "baseline.joblib"
INDEX_DIR = ARTIFACTS / "search_index"

# Models (choices and evidence in docs/DECISIONS.md)
CLASSIFIER_MODEL = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

CATEGORIES = ["ai", "app", "security"]
CATEGORY_AR = {"ai": "ذكاء اصطناعي", "app": "تطبيقات", "security": "أمن سيبراني"}

# Training
SEED = 42
TEST_SIZE = 0.25
LEARNING_RATE = 2e-5
EPOCHS = 10
BATCH_SIZE = 4
MAX_LENGTH = 128
