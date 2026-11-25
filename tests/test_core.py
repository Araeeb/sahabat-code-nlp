"""Golden tests for the parts that must never silently change."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sahabat.data import grouped_split, load_projects
from sahabat.entities import IGNORE, align_labels, extract_technologies, word_level_accuracy
from sahabat.evaluation import bootstrap_ci, recall_and_mrr
from sahabat.preprocessing import (CANARY_IN, CANARY_OUT, check_canary, mask_pii,
                                   normalize_arabic, preprocess)


# ---------------- preprocessing (R1)
def test_canary():
    check_canary()
    assert preprocess(CANARY_IN) == CANARY_OUT


def test_pii_masking_recall():
    cases = {
        "راسلني على sara@uni.edu.sa": "راسلني على [EMAIL]",
        "contact: team@gmail.com": "contact: [EMAIL]",
        "رقمي 0551234567": "رقمي [PHONE]",
        "call +966551234567 now": "call [PHONE] now",
        "رقم الشعبة 34556": "رقم الشعبة 34556",
    }
    for raw, expected in cases.items():
        assert mask_pii(raw) == expected


def test_arabic_normalisation():
    assert normalize_arabic("أداة إلكترونية على مستوى الجامعة") == "اداه الكترونيه علي مستوي الجامعه"
    assert normalize_arabic("الاختبـــار") == "الاختبار"
    assert normalize_arabic("آلياً") == "الياً"


def test_keeps_signal():
    assert "😍" in preprocess("رااائع 😍")
    assert "رااائع" in preprocess("رااائع 😍")
    assert "OpenCV" in preprocess("باستخدام OpenCV")


# ---------------- data split (leakage)
def test_no_team_leakage():
    train, test = grouped_split(load_projects())
    assert not set(train["team_id"]) & set(test["team_id"])
    assert len(train) + len(test) == len(load_projects())


# ---------------- entities (Lab 3B)
def test_align_labels_first_piece_only():
    word_ids = [None, 0, 1, 2, 2, 3, 4, 4, None]
    labels = ["O", "B-PER", "O", "O", "B-TECH"]
    assert align_labels(word_ids, labels) == [IGNORE, "O", "B-PER", "O", IGNORE, "O", "B-TECH", IGNORE, IGNORE]


def test_clitic_word_keeps_one_label():
    # "وسارة" split into two pieces must still be ONE B-PER
    assert align_labels([None, 0, 0, None], ["B-PER"]).count("B-PER") == 1


def test_extract_technologies():
    text = preprocess("تطبيق بـ Flutter و بايثون مع OpenCV")
    assert set(extract_technologies(text)) >= {"Flutter", "Python", "OpenCV"}


def test_word_level_accuracy():
    assert word_level_accuracy(["O", "B-PER"], ["O", "O"]) == 0.5


# ---------------- evaluation
def test_recall_and_mrr():
    recall, mrr = recall_and_mrr([[True, False], [False, True], [False, False]])
    assert recall == 2 / 3
    assert abs(mrr - (1 + 0.5 + 0) / 3) < 1e-9


def test_bootstrap_ci_contains_point():
    point, low, high = bootstrap_ci([0, 1, 2, 0, 1], [0, 1, 2, 1, 1])
    assert low <= point <= high
