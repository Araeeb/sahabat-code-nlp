"""Build the evaluation evidence from your own run (Labs 5, 6, 7).

Writes artifacts/BENCHMARKS.md and artifacts/evaluation.json.
Run after train.py:  python evaluate.py
"""
import io
import json
import time

import numpy as np
import pandas as pd

from sahabat import config
from sahabat.evaluation import bootstrap_ci, macro_f1, recall_and_mrr, sliced_accuracy
from sahabat.search import ProjectSearch


def classification_section(lines, results):
    preds = pd.read_csv(config.ARTIFACTS / "test_predictions.csv")
    score, low, high = bootstrap_ci(preds["label"], preds["pred"], metric=lambda a, b: macro_f1(a, b))
    results["test_macro_f1"] = {"value": round(score, 3), "ci95": [round(low, 3), round(high, 3)], "n": len(preds)}

    lines += ["## 1. التصنيف على مجموعة الاختبار المجمدة", "",
              f"- عدد الأوصاف: {len(preds)}",
              f"- Macro-F1: **{score:.3f}** (فترة ثقة 95%: {low:.3f} – {high:.3f})", ""]

    slices = sliced_accuracy(preds)
    results["slices"] = slices.to_dict(orient="records")
    lines += ["### التقييم حسب الشرائح", "", slices.to_markdown(index=False), ""]

    errors = preds[preds["label"] != preds["pred"]]
    lines += ["### الأخطاء (للقراءة اليدوية وتصنيف السبب)", ""]
    if errors.empty:
        lines += ["لا توجد أخطاء في مجموعة الاختبار.", ""]
    else:
        table = pd.DataFrame({"text": errors["text"],
                              "true": [config.CATEGORIES[i] for i in errors["label"]],
                              "pred": [config.CATEGORIES[i] for i in errors["pred"]],
                              "reason": ""})
        lines += [table.to_markdown(index=False), ""]
    results["n_errors"] = int(len(errors))
    return preds


def search_section(lines, results):
    search = ProjectSearch.load()
    queries = pd.read_csv(config.QUERIES_PATH)

    def evaluate(rerank):
        rel_by_lang = {"ar": [], "en": [], "all": []}
        for _, q in queries.iterrows():
            relevant = {int(t) for t in str(q["relevant_teams"]).split()}
            hits = search.search(q["query"], k=10, rerank=rerank)
            rel = [int(t) in relevant for t in hits["team_id"]]
            rel_by_lang[q["language"]].append(rel)
            rel_by_lang["all"].append(rel)
        return {lang: recall_and_mrr(r) for lang, r in rel_by_lang.items() if r}

    rows = []
    for name, rerank in [("FAISS only", False), ("FAISS + cross-encoder", True)]:
        res = evaluate(rerank)
        results[f"search_{'rerank' if rerank else 'faiss'}"] = res
        for lang, (rec, mrr) in res.items():
            rows.append({"stage": name, "queries": lang, "recall@10": round(rec, 2), "MRR@10": round(mrr, 2)})

    lines += ["## 2. البحث بالمعنى", "", f"عدد الأسئلة: {len(queries)}", "",
              pd.DataFrame(rows).to_markdown(index=False), "",
              "الفرق بين صفوف `ar` و `en` هو فجوة اللغة (cross-lingual gap).", ""]


def latency_section(lines, results, test_texts, test_labels):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (config.CLASSIFIER_DIR / "sahabat.json").exists():
        lines += ["## 3. السرعة", "", "تخطّينا القياس لأن الموديل المضبوط غير موجود (`--skip-bert`).", ""]
        return

    tok = AutoTokenizer.from_pretrained(str(config.CLASSIFIER_DIR))
    fp32 = AutoModelForSequenceClassification.from_pretrained(str(config.CLASSIFIER_DIR)).eval()
    int8 = torch.ao.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)
    bench = list(test_texts) * 20

    def size_mb(model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return round(buf.getbuffer().nbytes / 1e6, 1)

    def rung(name, model, pad_to=None):
        lat = []
        for t in bench:
            kw = dict(padding="max_length", max_length=pad_to) if pad_to else {}
            enc = tok(t, return_tensors="pt", truncation=True, **kw)
            start = time.perf_counter()
            with torch.no_grad():
                model(**enc)
            lat.append((time.perf_counter() - start) * 1000)
        preds = []
        for t in test_texts:
            with torch.no_grad():
                preds.append(int(model(**tok(t, return_tensors="pt", truncation=True)).logits.argmax(-1)))
        return {"rung": name, "p50 ms": round(float(np.percentile(lat, 50)), 1),
                "p99 ms": round(float(np.percentile(lat, 99)), 1),
                "macro-F1": round(macro_f1(test_labels, preds), 3), "size MB": size_mb(model)}

    torch.set_num_threads(max(1, torch.get_num_threads()))
    ladder = pd.DataFrame([rung("fp32 @512 padded", fp32, 512),
                           rung("fp32 dynamic padding", fp32),
                           rung("INT8 dynamic", int8)])
    results["latency_ladder"] = ladder.to_dict(orient="records")
    lines += ["## 3. السرعة على CPU (سلّم التحسين)", "", ladder.to_markdown(index=False), ""]


def main():
    results, lines = {}, ["# BENCHMARKS — أرقام من تشغيلنا", "",
                          "هذا الملف يتولد تلقائياً من `python evaluate.py`. لا تعدّل الأرقام يدوياً.", ""]
    train_metrics = json.loads((config.ARTIFACTS / "train_metrics.json").read_text())
    lines += ["## 0. خط الأساس مقابل الموديل المضبوط", "",
              f"- TF-IDF + LinearSVC: **{train_metrics['baseline_macro_f1']}**",
              f"- CAMeLBERT: **{train_metrics.get('camelbert_macro_f1', 'لم يُدرّب')}**",
              f"- الفرق: **{train_metrics.get('gain_points', '-')}** نقطة", ""]

    preds = classification_section(lines, results)
    search_section(lines, results)
    latency_section(lines, results, preds["clean_text"].tolist(), preds["label"].tolist())

    (config.ARTIFACTS / "BENCHMARKS.md").write_text("\n".join(lines), encoding="utf-8")
    (config.ARTIFACTS / "evaluation.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
