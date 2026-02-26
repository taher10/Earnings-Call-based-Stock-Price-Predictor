import streamlit as st
from pathlib import Path
import joblib

from data_cleaning import DataCleaning


RECOMMEND = {1: "BUY", 0: "SELL", 2: "HOLD"}


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def main():
    st.title("Earnings Transcript -> Buy/Sell Recommendation")
    st.write("Upload an earnings call transcript (plain text or .txt) to get a recommendation.")

    model_path = st.text_input("Model path", value="models/model.pkl")
    uploaded = st.file_uploader("Transcript file", type=["txt", "csv", "pdf"], accept_multiple_files=False)

    if st.button("Load Model"):
        if not Path(model_path).exists():
            st.error("Model not found at: " + model_path)
        else:
            st.success("Model loaded (will be used when predicting)")

    if uploaded is not None:
        raw = uploaded.read().decode(errors="ignore")
        st.subheader("Raw transcript (first 2k chars)")
        st.text(raw[:2000])

        dc = DataCleaning()
        clean = dc.clean_text(raw)

        if not Path(model_path).exists():
            st.error("Model not found. Please run training and provide the model path.")
            return

        model = load_model(model_path)
        probs = model.predict_proba([clean])[0]
        pred = int(model.predict([clean])[0])

        st.subheader("Recommendation")
        st.write(RECOMMEND.get(pred, "UNKNOWN"))
        st.subheader("Class probabilities")
        st.write({"sell": float(probs[0]), "buy": float(probs[1]) if len(probs)>1 else None, "hold": float(probs[2]) if len(probs)>2 else None})


if __name__ == '__main__':
    main()
