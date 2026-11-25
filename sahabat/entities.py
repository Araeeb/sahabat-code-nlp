"""Entity extraction (Lab 3B).

Two parts:
1. align_labels(): the subword label-alignment rule from Lab 3B — first piece of
   a word gets the word's tag, continuation pieces get -100 (ignored by loss and
   evaluation). Ready for training a full NER model (see docs, future work).
2. extract_technologies(): a transparent rule-based extractor the platform uses
   today, so the demo works without a trained NER model.
"""
import re

IGNORE = -100

# canonical name -> spellings to look for (already normalised by preprocess)
TECH_LEXICON = {
    "Python": ["python", "بايثون"],
    "Java": ["java", "جافا"],
    "Flutter": ["flutter", "فلاتر"],
    "React": ["react"],
    "OpenCV": ["opencv"],
    "BERT": ["bert"],
    "TensorFlow": ["tensorflow"],
    "PyTorch": ["pytorch"],
    "FAISS": ["faiss"],
    "QR": ["qr"],
    "CNN": ["cnn"],
    "Machine Learning": ["machine learning", "تعلم الاله"],
    "Deep Learning": ["deep learning", "التعلم العميق"],
    "AI": ["الذكاء الاصطناعي", "artificial intelligence"],
    "Chatbot": ["شات بوت", "chatbot", "بوت"],
    "Encryption": ["تشفير", "encrypt"],
    "Two-factor auth": ["two-factor", "تحقق بخطوتين"],
}


def extract_technologies(clean_text: str) -> list:
    lowered = clean_text.lower()
    found = []
    for name, spellings in TECH_LEXICON.items():
        for s in spellings:
            if re.search(rf"(?<![\w]){re.escape(s)}(?![\w])", lowered):
                found.append(name)
                break
    return found


def align_labels(word_ids, word_labels):
    """Map word-level tags onto subword tokens (first piece only)."""
    aligned, previous = [], None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(IGNORE)
        elif word_id != previous:
            aligned.append(word_labels[word_id])
        else:
            aligned.append(IGNORE)
        previous = word_id
    return aligned


def word_level_accuracy(y_true, y_pred):
    """Entity tags compared per WORD, never per subword."""
    return sum(t == p for t, p in zip(y_true, y_pred)) / max(len(y_true), 1)
