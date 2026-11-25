"""Project-category classifier (Lab 3A): TF-IDF baseline + fine-tuned CAMeLBERT."""
import json

import numpy as np

from . import config
from .preprocessing import PREPROC_VERSION


# ---------------------------------------------------------------- baseline
def train_baseline(train_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_pipeline
    from sklearn.svm import LinearSVC

    model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)), LinearSVC())
    model.fit(train_df["clean_text"], train_df["label"])
    return model


# ---------------------------------------------------------------- fine-tuning
def fine_tune(train_df, eval_df=None, output_dir=config.CLASSIFIER_DIR):
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

    set_seed(config.SEED)
    tokenizer = AutoTokenizer.from_pretrained(config.CLASSIFIER_MODEL)

    def to_dataset(frame):
        enc = tokenizer(frame["clean_text"].tolist(), truncation=True, max_length=config.MAX_LENGTH)
        enc["labels"] = frame["label"].tolist()
        return Dataset.from_dict(dict(enc))

    model = AutoModelForSequenceClassification.from_pretrained(
        config.CLASSIFIER_MODEL,
        num_labels=len(config.CATEGORIES),
        id2label=dict(enumerate(config.CATEGORIES)),
        label2id={c: i for i, c in enumerate(config.CATEGORIES)},
    )

    args = TrainingArguments(
        output_dir=str(config.ARTIFACTS / "runs"),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        weight_decay=0.01,
        seed=config.SEED,
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=to_dataset(train_df),
        data_collator=DataCollatorWithPadding(tokenizer),   # dynamic padding (Lab 2 lesson)
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "sahabat.json").write_text(json.dumps(
        {"preproc_version": PREPROC_VERSION, "base_model": config.CLASSIFIER_MODEL,
         "categories": config.CATEGORIES, "seed": config.SEED}, ensure_ascii=False, indent=2))
    return output_dir


# ---------------------------------------------------------------- inference
class Classifier:
    """Loads the fine-tuned model saved by train.py."""

    def __init__(self, model_dir=config.CLASSIFIER_DIR, device=None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((model_dir / "sahabat.json").read_text())
        if meta["preproc_version"] != PREPROC_VERSION:
            raise RuntimeError(f"Train/serve skew: model trained with preprocessing "
                               f"{meta['preproc_version']}, code is {PREPROC_VERSION}. Retrain.")

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).to(self.device).eval()

    def predict_proba(self, clean_texts):
        enc = self.tokenizer(list(clean_texts), padding=True, truncation=True,
                             max_length=config.MAX_LENGTH, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**enc).logits
        return self.torch.softmax(logits, dim=-1).cpu().numpy()

    def predict(self, clean_texts):
        return np.argmax(self.predict_proba(clean_texts), axis=-1)
