# 🎬 Movie Review Sentiment Analysis using DistilBERT

A Natural Language Processing (NLP) project that classifies movie reviews as **Positive** or **Negative** using a fine-tuned DistilBERT model.

The project covers the complete workflow from text preprocessing and tokenization to model fine-tuning, evaluation, error analysis, and deployment using Streamlit.

---

## 🚀 Live Demo

### 🌐 [Try the Movie Review Sentiment Analyzer](YOUR_STREAMLIT_APP_LINK)

Enter your own movie review and get a **Positive** or **Negative** sentiment prediction with confidence scores.

---

## 📌 Project Overview

Sentiment Analysis is an NLP task used to determine the sentiment expressed in a piece of text.

In this project, movie reviews are classified into two categories:

- 🟢 Positive
- 🔴 Negative

The project was developed to understand how Transformer-based models can be applied to real-world NLP classification problems.

Before using DistilBERT, LSTM and BiLSTM models were also implemented and evaluated as part of the experimentation process.

---

## 📊 Dataset

The project uses the **IMDb Dataset of 50K Movie Reviews**.

The dataset contains 50,000 movie reviews with two sentiment classes:

- Positive: 25,000 reviews
- Negative: 25,000 reviews

### Dataset Split

| Dataset | Number of Reviews |
|---------|------------------:|
| Training | 40,000 |
| Validation | 5,000 |
| Testing | 5,000 |
| **Total** | **50,000** |

The dataset is balanced, with an equal number of positive and negative reviews.

### Dataset Source

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

---

# 🔄 Project Workflow

```text
IMDb Movie Reviews
        ↓
Text Cleaning
        ↓
Train / Validation / Test Split
        ↓
DistilBERT Tokenization
        ↓
Pretrained DistilBERT
        ↓
Fine-Tuning
        ↓
Validation
        ↓
Testing
        ↓
Error Analysis
        ↓
Model Saving
        ↓
Streamlit Deployment
```

---

# 🧹 Text Preprocessing

The raw IMDb reviews contain HTML tags such as `<br />`.

Basic cleaning was performed before passing the reviews to the tokenizer.

The preprocessing included:

- Removing HTML line-break tags
- Removing unnecessary whitespace
- Cleaning the text
- Preserving the original sentiment-related words

Stopword removal was not applied because words such as **"not"**, **"never"**, and similar words can be important for sentiment classification.

---

# 🔤 DistilBERT Tokenization

Instead of manually creating a vocabulary, the pretrained DistilBERT tokenizer was used.

The tokenizer converts the review into:

- Input IDs
- Attention masks

The maximum sequence length was set to **256 tokens**.

Longer reviews were truncated, while shorter reviews were padded.

The attention mask allows the model to distinguish actual tokens from padding tokens.

---

# 🤖 Model Architecture

The final model uses:

**DistilBERT (`distilbert-base-uncased`)**

DistilBERT is a smaller and faster Transformer-based language model derived from BERT while retaining much of its language understanding capability.

The pretrained model was fine-tuned for binary sentiment classification.

### Main Architecture

```text
Input Review
     ↓
DistilBERT Tokenizer
     ↓
Input IDs + Attention Mask
     ↓
DistilBERT
     ↓
Classification Head
     ↓
2 Output Classes
     ↓
Negative / Positive
```

The classification head produces two logits:

```text
Class 0 → Negative
Class 1 → Positive
```

Softmax is then used to obtain the probabilities for the two classes.

---

# ⚙️ Training

The pretrained DistilBERT model was fine-tuned using PyTorch.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | DistilBERT |
| Maximum Sequence Length | 256 |
| Batch Size | 16 |
| Optimizer | AdamW |
| Learning Rate | 2 × 10⁻⁵ |
| Loss Function | Cross Entropy Loss |
| Device | NVIDIA Tesla T4 GPU |
| Number of Epochs | 3 |

Mixed-precision training was used to improve GPU efficiency and reduce training time.

---

# 📈 Model Performance

Three models were experimented with during the project.

| Model | Validation Accuracy | Test Accuracy |
|-------|---------------------:|--------------:|
| LSTM | 87.96% | **87.94%** |
| BiLSTM | 86.72% | **86.46%** |
| DistilBERT | 92.24% | **92.28%** |

### Final DistilBERT Test Accuracy

# **92.28%**

The final result was obtained on **5,000 previously unseen IMDb test reviews**.

---

# 🧪 Evaluation

The model was evaluated using the held-out test dataset.

Evaluation focused on:

- Accuracy
- Prediction probabilities
- Correct and incorrect predictions
- Manual testing
- Error analysis

Additional manually written reviews were also used to investigate how the model behaves on challenging examples.

---

# 🔍 Error Analysis

The model performed well on the IMDb test dataset but was also tested using deliberately challenging reviews.

These reviews included:

- Negation
- Contrasting opinions
- Positive and negative statements in the same review
- Long-range sentiment changes
- Subtle sentiment expressions

For example, a review can contain many positive statements about the acting, cinematography, and soundtrack while ultimately expressing dissatisfaction with the movie.

These examples showed that even a high-performing sentiment classifier can struggle when the overall sentiment depends on complex context.

This was an important part of the project because it demonstrated that benchmark accuracy does not guarantee perfect performance on every real-world example.

---

# 🆚 Model Comparison

LSTM and BiLSTM models were first implemented to understand sequence-based sentiment classification.

DistilBERT was then explored to understand Transformer-based NLP and pretrained language models.

| Model | Approach | Test Accuracy |
|-------|----------|--------------:|
| LSTM | Recurrent Neural Network | **87.94%** |
| BiLSTM | Bidirectional Recurrent Neural Network | **86.46%** |
| DistilBERT | Transformer | **92.28%** |

DistilBERT achieved the highest test accuracy among the models tested and was selected as the final model for deployment.

---

# 🌐 Streamlit Deployment

The trained DistilBERT model was deployed using **Streamlit**.

The application allows users to enter their own movie reviews and receive a sentiment prediction.

### Application Features

- 🎬 Movie review input
- 🟢 Positive prediction
- 🔴 Negative prediction
- 📊 Confidence score
- 📈 Prediction probability breakdown

---

# 🖥️ Running the Application Locally

### 1. Clone the repository

```bash
git clone https://github.com/purple279/Sentiment_Analysis_DistilBERT.git
cd Sentiment_Analysis_DistilBERT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 📁 Project Structure

```text
Sentiment-Analysis/
│
├── app.py
│
├── Models/
│   └── DistilBERT/
│       ├── model/
│       │   ├── config.json
│       │   └── model.safetensors
│       │
│       └── tokenizer/
│           ├── tokenizer_config.json
│           └── tokenizer.json
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technologies Used

- **Python**
- **PyTorch**
- **Hugging Face Transformers**
- **DistilBERT**
- **Scikit-learn**
- **Streamlit**
- **Kaggle**
- **Jupyter Notebook**

---

# 📚 Key Concepts Learned

Through this project, I gained practical experience with:

- Natural Language Processing
- Text preprocessing
- Tokenization
- Attention masks
- Transformer architecture
- BERT and DistilBERT
- Transfer learning
- Fine-tuning pretrained models
- PyTorch model training
- GPU-based training
- Model evaluation
- Error analysis
- Model saving and loading
- Streamlit deployment

---

# 💡 Key Learnings

This project helped me understand the progression from recurrent neural networks to Transformer-based NLP.

I first implemented LSTM and BiLSTM models to understand how recurrent neural networks process text sequences.

I then moved to DistilBERT to understand how pretrained Transformer models can be fine-tuned for downstream NLP tasks.

One of the most important lessons from the project was that a strong test accuracy does not mean that a model will correctly classify every possible real-world example. Testing challenging reviews helped identify limitations involving negation, contrast, and subtle sentiment changes.

---

# 🔮 Future Improvements

Possible future improvements include:

- More extensive error analysis
- Hyperparameter experimentation
- Testing on additional sentiment datasets
- Improving robustness to negation and contrasting sentiment
- Exploring larger Transformer models
- Further optimization for production deployment

---

# 📄 License

This project is licensed under the MIT License.

---

# 👩‍💻 Author

**Vashundthera**

GitHub: https://github.com/purple279
