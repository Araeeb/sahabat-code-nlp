"""Train every Sahabat Code component and save it to artifacts/.

    python train.py              # baseline + fine-tuned CAMeLBERT + search index
    python train.py --skip-bert  # baseline + search index only (fast, CPU)
"""
import argparse
import json

import joblib

from sahabat import config
from sahabat.classifier import Classifier, fine_tune, train_baseline
from sahabat.data import grouped_split, load_projects
from sahabat.evaluation import macro_f1
from sahabat.preprocessing import PREPROC_VERSION, check_canary
from sahabat.search import ProjectSearch


def main(skip_bert=False):
    check_canary()
    config.ARTIFACTS.mkdir(exist_ok=True)

    df = load_projects()
    train_df, test_df = grouped_split(df)
    print(f"Data: {len(df)} projects | train {len(train_df)} | frozen test {len(test_df)} | shared teams: none")
    test_df[["project_id"]].to_csv(config.ARTIFACTS / "test_ids.csv", index=False)

    # 1) baseline
    baseline = train_baseline(train_df)
    joblib.dump(baseline, config.BASELINE_PATH)
    test_df["pred_baseline"] = baseline.predict(test_df["clean_text"])
    metrics = {"preproc_version": PREPROC_VERSION, "n_train": len(train_df), "n_test": len(test_df),
               "baseline_macro_f1": round(macro_f1(test_df["label"], test_df["pred_baseline"]), 3)}
    print("TF-IDF + LinearSVC macro-F1:", metrics["baseline_macro_f1"])

    # 2) fine-tuned CAMeLBERT
    if not skip_bert:
        fine_tune(train_df)
        clf = Classifier()
        test_df["pred"] = clf.predict(test_df["clean_text"])
        metrics["camelbert_macro_f1"] = round(macro_f1(test_df["label"], test_df["pred"]), 3)
        metrics["gain_points"] = round(100 * (metrics["camelbert_macro_f1"] - metrics["baseline_macro_f1"]), 1)
        print("CAMeLBERT macro-F1:", metrics["camelbert_macro_f1"], "| gain:", metrics["gain_points"], "points")
    else:
        test_df["pred"] = test_df["pred_baseline"]

    test_df.to_csv(config.ARTIFACTS / "test_predictions.csv", index=False)

    # 3) search index over the whole catalogue
    manifest = ProjectSearch().build(df).save()
    print("Search index:", manifest["size"], "projects | manifest saved")

    (config.ARTIFACTS / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Done. Artifacts in", config.ARTIFACTS)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true")
    main(parser.parse_args().skip_bert)
