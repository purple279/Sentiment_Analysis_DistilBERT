import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #888888;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .result {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
        margin-top: 25px;
        margin-bottom: 15px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATH = "Models/DistilBERT/model"
TOKENIZER_PATH = "Models/DistilBERT/tokenizer"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_PATH
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    )

    model.eval()

    return tokenizer, model


tokenizer, model = load_model()


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Find out whether a movie review is positive or negative'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# REVIEW INPUT
# ============================================================

review = st.text_area(
    "Your Review",
    placeholder="Write your movie review here...",
    height=180,
    label_visibility="collapsed"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True
):

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        # ----------------------------------------------------
        # TOKENIZE
        # ----------------------------------------------------

        inputs = tokenizer(
            review,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )

        # DistilBERT does not use token_type_ids
        inputs.pop("token_type_ids", None)


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )


        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        negative_probability = (
            probabilities[0][0].item() * 100
        )

        positive_probability = (
            probabilities[0][1].item() * 100
        )


        # ----------------------------------------------------
        # SENTIMENT
        # ----------------------------------------------------

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()


        if predicted_class == 1:

            sentiment = "Positive"
            confidence = positive_probability

            st.markdown(
                '<div class="result">🟢 Positive</div>',
                unsafe_allow_html=True
            )

        else:

            sentiment = "Negative"
            confidence = negative_probability

            st.markdown(
                '<div class="result">🔴 Negative</div>',
                unsafe_allow_html=True
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.subheader("Confidence")

        st.progress(int(confidence))

        st.write(
            f"**{confidence:.2f}%**"
        )


        # ----------------------------------------------------
        # PROBABILITY BREAKDOWN
        # ----------------------------------------------------

        st.subheader("Prediction Breakdown")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "🔴 Negative",
                f"{negative_probability:.2f}%"
            )

        with col2:

            st.metric(
                "🟢 Positive",
                f"{positive_probability:.2f}%"
            )