"""Classify a news article from the command line.
Usage: python predict.py "article text"   |   python predict.py -f article.txt
"""
import argparse
import joblib
import numpy as np
from train import clean_text, MODEL_PATH


def predict(text: str, model=None, top_k: int = 5):
    model = model or joblib.load(MODEL_PATH)
    x = clean_text(text)
    clf = model.named_steps["clf"]
    if hasattr(clf, "predict_proba"):
        p_fake = float(model.predict_proba([x])[0][1])
    else:  # PassiveAggressive: squash the margin into 0-1
        p_fake = float(1 / (1 + np.exp(-model.decision_function([x])[0])))
    label = "FAKE" if p_fake >= 0.5 else "REAL"

    words = []
    if hasattr(clf, "coef_"):
        vec = model.named_steps["tfidf"]
        vocab = np.array(vec.get_feature_names_out())
        row = vec.transform([x])
        contrib = row.multiply(clf.coef_[0]).toarray()[0]
        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        words = [(vocab[i], float(contrib[i])) for i in idx if contrib[i] != 0]
    return label, p_fake, words


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?", help="article text")
    ap.add_argument("-f", "--file", help="read article from a text file")
    a = ap.parse_args()
    text = open(a.file, encoding="utf-8").read() if a.file else a.text
    if not text:
        ap.error("provide text or --file")
    label, p, words = predict(text)
    print(f"Prediction: {label}  (P(fake) = {p:.1%})")
    for w, c in words:
        print(f"  {'->FAKE' if c > 0 else '->REAL'}  {w:20s} {c:+.3f}")
