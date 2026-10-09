"""Web UI.  Run: streamlit run app.py"""
import joblib
import streamlit as st
from predict import predict
from train import MODEL_PATH

st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.caption("A TF-IDF + machine-learning classifier. It is a decision aid, "
           "not a fact-checker - always verify important claims.")

@st.cache_resource
def load():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("No model found. Run `python train.py` first.")
    st.stop()

text = st.text_area("Paste a news headline or article", height=250)
if st.button("Analyze") and text.strip():
    label, p, words = predict(text, load())
    (st.error if label == "FAKE" else st.success)(
        f"Prediction: **{label}** - P(fake) = {p:.1%}")
    st.progress(p)
    if words:
        st.subheader("Most influential words")
        for w, c in words:
            st.write(f"{'🔴' if c > 0 else '🟢'} `{w}` ({c:+.3f})")
