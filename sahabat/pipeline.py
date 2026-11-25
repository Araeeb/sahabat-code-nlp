"""The integrated Sahabat Code service: every lab component in one call.

    preprocess (Labs 1, 4) -> classify (Lab 3A) -> extract (Lab 3B) -> similar projects (Lab 5)
"""
from . import config
from .entities import extract_technologies
from .preprocessing import check_canary, count_pii, preprocess


class SahabatPipeline:
    def __init__(self, load_classifier=True, load_search=True):
        check_canary()                                   # skew canary at startup
        self.classifier = None
        self.baseline = None
        self.search = None

        if load_classifier and (config.CLASSIFIER_DIR / "sahabat.json").exists():
            from .classifier import Classifier
            self.classifier = Classifier()
        elif config.BASELINE_PATH.exists():
            import joblib
            self.baseline = joblib.load(config.BASELINE_PATH)
            print("Fine-tuned model not found, using the TF-IDF baseline.")
        else:
            raise FileNotFoundError("No trained model. Run: python train.py")

        if load_search and (config.INDEX_DIR / "manifest.json").exists():
            from .search import ProjectSearch
            self.search = ProjectSearch.load()

    def classify(self, clean_text):
        if self.classifier is not None:
            probs = self.classifier.predict_proba([clean_text])[0]
            best = int(probs.argmax())
            return config.CATEGORIES[best], float(probs[best])
        return config.CATEGORIES[int(self.baseline.predict([clean_text])[0])], None

    def analyze(self, project_text, k=3):
        clean = preprocess(project_text)
        category, confidence = self.classify(clean)
        result = {
            "clean_text": clean,
            "category": category,
            "category_ar": config.CATEGORY_AR[category],
            "confidence": None if confidence is None else round(confidence, 3),
            "technologies": extract_technologies(clean),
            "pii_masked": count_pii(project_text),
            "similar_projects": [],
        }
        if self.search is not None:
            hits = self.search.search(project_text, k=k)
            result["similar_projects"] = [
                {"text": t, "category": c, "score": round(float(s), 3)}
                for t, c, s in zip(hits["text"], hits["category"], hits["score"])]
        return result
