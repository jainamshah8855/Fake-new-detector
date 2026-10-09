"""Train and compare fake-news classifiers (TF-IDF + linear models)."""
import re
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier, PassiveAggressiveClassifier
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_DIR = Path("data")
MODEL_PATH = Path("model.joblib")


def clean_text(text: str) -> str:
    text = str(text).lower()
    # Remove source tags like "WASHINGTON (Reuters) -" that leak the label
    text = re.sub(r"^.{0,60}\(reuters\)\s*-?\s*", "", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sample_data() -> pd.DataFrame:
    """Tiny synthetic fallback so the pipeline can be smoke-tested."""
    fake = ["shocking secret cure doctors hate revealed miracle",
            "celebrity clone caught aliens government hiding truth",
            "you won't believe this one weird trick banks fear",
            "breaking exposed hoax elites control weather machine"]
    real = ["parliament passes budget bill after lengthy debate",
            "central bank holds interest rates steady this quarter",
            "researchers publish peer reviewed study on climate data",
            "city council approves new public transport funding plan"]
    import random
    random.seed(0)
    filler = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
              "kilo lima mike november oscar papa quebec romeo sierra tango").split()
    mk = lambda t: t + " " + " ".join(random.sample(filler, 4))
    rows = [(mk(t), 1) for _ in range(25) for t in fake]
    rows += [(mk(t), 0) for _ in range(25) for t in real]
    return pd.DataFrame(rows, columns=["content", "label"])


def load_data() -> pd.DataFrame:
    fake_p, true_p = DATA_DIR / "Fake.csv", DATA_DIR / "True.csv"
    if not (fake_p.exists() and true_p.exists()):
        print("[!] data/Fake.csv and data/True.csv not found -> using tiny sample "
              "data. Download the real dataset (see README) for real results.\n")
        return sample_data()
    fake, true = pd.read_csv(fake_p), pd.read_csv(true_p)
    fake["label"], true["label"] = 1, 0          # 1 = FAKE, 0 = REAL
    df = pd.concat([fake, true], ignore_index=True)
    df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["content", "label"]]


def main():
    df = load_data()
    df["content"] = df["content"].map(clean_text)
    df = df[df["content"].str.len() > 0].drop_duplicates("content")
    print(f"Samples: {len(df)} | fake share: {df['label'].mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=42)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "PassiveAggressive": SGDClassifier(loss="hinge", penalty=None, learning_rate="pa1",
                                           eta0=1.0, max_iter=50, random_state=42),
        "MultinomialNB": MultinomialNB(),
    }
    best_name, best_pipe, best_f1 = None, None, -1
    for name, clf in models.items():
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_df=0.7,
                                      min_df=2 if len(df) > 500 else 1,
                                      ngram_range=(1, 2), max_features=100_000)),
            ("clf", clf),
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        f1 = f1_score(y_test, pred)
        print(f"{name:20s} acc={accuracy_score(y_test, pred):.4f}  F1={f1:.4f}")
        if f1 > best_f1:
            best_name, best_pipe, best_f1 = name, pipe, f1

    print(f"\nBest model: {best_name}\n")
    pred = best_pipe.predict(X_test)
    print(classification_report(y_test, pred, target_names=["REAL", "FAKE"], zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["REAL", "FAKE"], cmap="Blues")
    plt.title(f"Confusion matrix - {best_name}")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(best_pipe, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    sys.exit(main())
