# Fake News Detector (AI / NLP)

Classifies news articles as **REAL** or **FAKE** using TF-IDF features and
classical ML models (Logistic Regression, Passive Aggressive, Naive Bayes).
The best model by F1 score is saved automatically.

## Setup
```bash
pip install -r requirements.txt
```

## Dataset
Download the *Fake and Real News Dataset* (ISOT) from Kaggle and place
`Fake.csv` and `True.csv` inside `data/`. Without them, `train.py` runs on a tiny
synthetic sample only to prove the pipeline works.

## Run
```bash
python train.py                       # trains, evaluates, saves model.joblib
python predict.py "Your article text" # CLI prediction with key words
streamlit run app.py                  # web interface
```

## How it works
1. Merge title + body, lowercase, strip URLs/punctuation/source tags.
2. TF-IDF with unigrams + bigrams, English stop-words removed.
3. Train 3 classifiers, compare accuracy/F1, keep the best.
4. Explain predictions via the top weighted words (linear models).

## Limitations (important for your report)
- The ISOT dataset has stylistic leakage (e.g. "(Reuters)"); accuracy of 98-99%
  overstates real-world performance. We strip such tags, but bias remains.
- The model detects *writing style*, not factual truth.
- Fails on new topics/time periods (concept drift) - retrain regularly.

## Ideas to extend
- Fine-tune BERT/DistilBERT (HuggingFace) for higher accuracy.
- Add source-credibility and fact-check API features.
- Evaluate on a different dataset (e.g. LIAR, FakeNewsNet) to test generalization.
