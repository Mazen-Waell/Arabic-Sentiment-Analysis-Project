# Arabic Sentiment Analysis 🔍

A production-ready NLP system for analyzing sentiment in Arabic text. Built with fine-tuned AraBERT on 99K Arabic reviews.

> **Live Demo:** [huggingface.co/spaces/MazenWael/Rabic-Sentiment-Analysis](https://huggingface.co/spaces/MazenWael/Rabic-Sentiment-Analysis)  
> **Model:** [huggingface.co/MazenWael/arabert-sentiment](https://huggingface.co/MazenWael/arabert-sentiment)

---

## 📌 Problem Statement

Egyptian companies receive thousands of Arabic customer reviews daily across social media, Amazon, and Noon — making manual analysis impossible at scale. This system automatically classifies each comment as **Positive**, **Negative**, or **Neutral** with a confidence score.

---

## 🗂️ Dataset

**Arabic 100K Reviews** — 99,417 reviews from hotels, books, movies, and products.  
Source: [kaggle.com/datasets/abedkhooli/arabic-100k-reviews](https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews)

| Split | Size |
|-------|------|
| Train | 79,533 |
| Test  | 19,884 |

**Label distribution (balanced):**

| Label | Count |
|-------|-------|
| Neutral  | 33,179 |
| Positive | 33,160 |
| Negative | 33,078 |

**Preprocessing steps:**
1. Removed URLs, mentions, hashtags
2. Kept Arabic characters only (`\u0600-\u06FF`)
3. Removed texts shorter than 10 characters
4. Replaced `Mixed` label with `Neutral`

---

## 🏗️ Model Comparison

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| TF-IDF + Logistic Regression (baseline) | 66% | 0.66 |
| **AraBERT fine-tuned (ours)** | **74%** | **0.74** |

**Improvement: +8% F1 over baseline**

---

## 📊 Results

### Classification Report (AraBERT)

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| Negative | 0.80 | 0.78 | 0.79 |
| Neutral  | 0.64 | 0.64 | 0.64 |
| Positive | 0.77 | 0.79 | 0.78 |
| **Macro avg** | **0.74** | **0.74** | **0.74** |

### Training Config

| Hyperparameter | Value |
|----------------|-------|
| Model | cardiffnlp/twitter-xlm-roberta-base-sentiment |
| Epochs | 2 |
| Learning rate | 1e-5 |
| Batch size | 16 |
| Max length | 128 |
| Optimizer | AdamW |

### Error Analysis

- **Neutral** is the hardest class (F1: 0.64) — reviews with mixed opinions are ambiguous even for humans
- Model struggles with **Egyptian dialect negation** — e.g. "مش وحش" (means good) is misclassified as Negative
- Best performance on **Negative** class (F1: 0.79)

---

## ⚠️ Limitations

- Model performs best on Modern Standard Arabic (MSA)
- Egyptian and Levantine dialects may produce inaccurate results
- Slang and negation in dialects (e.g. "مش وحش") are not well-handled
- Training data is domain-specific (hotels, books, products)

---

## 🚀 API Usage

```
POST /predict
Content-Type: application/json
```

**Request:**
```json
{
  "text": "الفندق كان رائع جداً والخدمة ممتازة"
}
```

**Response:**
```json
{
  "text": "الفندق كان رائع جداً والخدمة ممتازة",
  "sentiment": "Positive",
  "confidence": 0.915,
  "scores": {
    "Negative": 0.032,
    "Neutral": 0.053,
    "Positive": 0.915
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Model | AraBERT / XLM-RoBERTa (HuggingFace) |
| Training | PyTorch + HuggingFace Transformers |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerization | Docker + Docker Compose |
| Deployment | Hugging Face Spaces |

---

## 📁 Project Structure

```
arabic-sentiment-analysis/
│
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned & labeled data
│
├── notebooks/
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_baseline.ipynb       # TF-IDF + Logistic Regression
│   └── 03_arabert_final.ipynb  # AraBERT fine-tuning
│
├── src/
│   └── preprocessing.py        # Arabic text cleaning pipeline
│
├── api/
│   └── main.py                 # FastAPI app
│
├── app/
│   └── streamlit_app.py        # Streamlit UI
│
├── models/
│   └── arabert-sentiment/      # Fine-tuned model
│
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## ⚙️ How to Run

### Option 1 — Docker Compose (recommended)

```bash
git clone https://github.com/Mazen-Waell/arabic-sentiment-analysis
cd arabic-sentiment-analysis
docker-compose up
```

- API: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

### Option 2 — Manual

```bash
pip install -r requirements.txt
python src/preprocessing.py
uvicorn api.main:app --reload
streamlit run app/streamlit_app.py
```

---

## 📈 Future Work

- [ ] Add Egyptian dialect training data
- [ ] Aspect-based sentiment (product quality vs. shipping vs. price)
- [ ] Support for more Arabic dialects (Levantine, Gulf)
- [ ] Real-time monitoring dashboard

---

## 👤 Author

**Mazen Wael** — [@Mazen-Waell](https://github.com/Mazen-Waell)

Alexandria University — Faculty of Engineering  
Computer & Communication Engineering, Class of 2027

---


