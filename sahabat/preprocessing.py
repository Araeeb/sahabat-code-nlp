"""One preprocessing truth (Labs 1 and 4).

Training, evaluation, the demo app and the API all import THIS module, so
text is always cleaned the same way. Bump PREPROC_VERSION on any change:
the pipeline refuses to start if the saved model used another version.

What it deliberately does NOT do: lowercase, strip emoji, or remove repeated
letters. Those carry meaning (tone, technical terms).
"""
import re
import unicodedata

PREPROC_VERSION = "1.0.0"

_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(r"(?:\+?966|\b0)5\d{8}\b")      # Saudi mobile numbers
_TATWEEL = "ـ"
_SPACES_RE = re.compile(r"\s+")

# Fixed input/output pair checked at startup (skew canary)
CANARY_IN = "  أريد   منصة إلكترونية على مستوى الجامعة، راسلني على test@uni.edu.sa أو 0551234567  "
CANARY_OUT = "اريد منصه الكترونيه علي مستوي الجامعه، راسلني علي [EMAIL] او [PHONE]"


def mask_pii(text: str) -> str:
    """Replace e-mails and Saudi mobile numbers with fixed tags."""
    text = _EMAIL_RE.sub("[EMAIL]", text)
    return _PHONE_RE.sub("[PHONE]", text)


def count_pii(text: str) -> dict:
    """How many e-mails and phone numbers a text contains (before masking)."""
    return {"emails": len(_EMAIL_RE.findall(text)), "phones": len(_PHONE_RE.findall(text))}


def normalize_arabic(text: str) -> str:
    """Unify Arabic letters that are written in more than one way."""
    text = unicodedata.normalize("NFC", text)      # compose letters + marks first
    text = text.replace(_TATWEEL, "")
    text = re.sub(r"[إأآ]", "ا", text)
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def normalize_spaces(text: str) -> str:
    return _SPACES_RE.sub(" ", text).strip()


def preprocess(text) -> str:
    """The only function the rest of the project calls.

    Order matters: mask PII BEFORE normalising, so the normaliser can never
    break an e-mail or phone pattern.
    """
    text = mask_pii(str(text))
    text = normalize_arabic(text)
    return normalize_spaces(text)


def check_canary() -> None:
    """Fail loudly if preprocessing changed without a version bump."""
    got = preprocess(CANARY_IN)
    if got != CANARY_OUT:
        raise RuntimeError(f"Preprocessing canary failed.\nexpected: {CANARY_OUT}\ngot:      {got}")
