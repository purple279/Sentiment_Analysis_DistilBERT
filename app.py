import streamlit as st
import torch
import torch.nn as nn
import pickle
import os


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Movie Review Sentiment",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# Custom UI
# --------------------------------------------------

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


# --------------------------------------------------
# Paths
# --------------------------------------------------

MODEL_PATH = "Models/LSTM/model.pth"
WORD_TO_INT_PATH = "Models/LSTM/word_to_int.pkl"
CONFIG_PATH = "Models/LSTM/config.pkl"


# --------------------------------------------------
# LSTM Model
# --------------------------------------------------

class SentimentLSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_size):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        x = self.embedding(x)

        output, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]

        x = self.fc(last_hidden)

        return x


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def load_model():

    # Load vocabulary
    with open(WORD_TO_INT_PATH, "rb") as f:
        word_to_int = pickle.load(f)

    # Load configuration
    with open(CONFIG_PATH, "rb") as f:
        config = pickle.load(f)

    # Get model parameters
    vocab_size = config["vocab_size"]
    embedding_dim = config["embedding_dim"]
    hidden_size = config["hidden_size"]

    # Create model
    model = SentimentLSTM(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size
    )

    # Load trained weights
    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=torch.device("cpu")
        )
    )

    model.eval()

    return model, word_to_int, config


model, word_to_int, config = load_model()


# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🎬 Movie Review Sentiment</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Find out whether a movie review is positive or negative</div>',
    unsafe_allow_html=True
)


# --------------------------------------------------
# Review input
# --------------------------------------------------

review = st.text_area(
    "Your Review",
    placeholder="Write your movie review here...",
    height=180,
    label_visibility="collapsed"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Analyze Sentiment", use_container_width=True):

    if not review.strip():

        st.warning("Please enter a movie review.")

    else:

        # ------------------------------------------
        # Preprocessing
        # ------------------------------------------

        review = review.replace("<br />", "")
        review = review.lower()

        tokens = review.split()

        # Convert words to integer IDs
        token_ids = []

        for word in tokens:

            if word in word_to_int:
                token_ids.append(word_to_int[word])
            else:
                token_ids.append(1)  # <UNK>


        # ------------------------------------------
        # Padding / truncation
        # ------------------------------------------

        max_len = config["max_len"]

        if len(token_ids) > max_len:

            token_ids = token_ids[:max_len]

        elif len(token_ids) < max_len:

            token_ids += [0] * (max_len - len(token_ids))


        # ------------------------------------------
        # Convert to tensor
        # ------------------------------------------

        input_tensor = torch.tensor(
            [token_ids],
            dtype=torch.long
        )


        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        with torch.no_grad():

            output = model(input_tensor)

            probability = torch.sigmoid(output)

            positive_probability = probability.item() * 100

            negative_probability = 100 - positive_probability


        # ------------------------------------------
        # Final sentiment
        # ------------------------------------------

        if positive_probability >= 50:

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


        # ------------------------------------------
        # Confidence
        # ------------------------------------------

        st.subheader("Confidence")

        st.progress(
            int(confidence)
        )

        st.write(
            f"**{confidence:.2f}%**"
        )


        # ------------------------------------------
        # Prediction breakdown
        # ------------------------------------------

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