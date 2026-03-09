# TruthScan — AI Fake News Detection System

> Detect fake vs real news using NLP + Machine Learning + AI-powered web interface

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Scikit-learn](https://img.shields.io/badge/ML-Scikit--learn-orange)
![NLP](https://img.shields.io/badge/NLP-TF--IDF-green)
![HTML](https://img.shields.io/badge/UI-HTML%2FCSS%2FJS-red)

---

## 📌 Overview

TruthScan is a full-stack fake news detection system combining classical NLP machine learning with a modern AI-powered web interface. It classifies news headlines and articles as **REAL** or **FAKE** with confidence scores, probability meters, and AI reasoning.

---

## 📁 Project Structure
```
fake-news-detector/
├── fake_news_detector.py     # ML backend (Python + Scikit-learn)
├── fake_news_website.html    # Frontend website (single HTML file)
├── requirements.txt          # Python dependencies
├── fake_news_model.pkl       # Saved model (auto-generated)
├── Fake.csv                  # Optional: Kaggle dataset
└── True.csv                  # Optional: Kaggle dataset
```

---

## ✨ Features

- ✅ Detects fake vs real news with 98%+ accuracy
- 🤖 3 ML models: Logistic Regression, Naive Bayes, Random Forest
- 📊 TF-IDF vectorization with unigrams + bigrams
- 🌐 Single-file website — no server needed
- 🧠 AI reasoning explains every prediction
- 💾 Model saved to `.pkl` — no retraining needed
- 🔌 Offline fallback heuristics if API unavailable

---

## 🚀 Quick Start

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the ML model**
```bash
python fake_news_detector.py
```

**3. Open the website**

Double-click `fake_news_website.html` in your browser — done!

---

## 📊 Dataset

| Mode | Description | Accuracy |
|------|-------------|----------|
| Synthetic (default) | Auto-generated, no download needed | ~85–90% |
| Kaggle Dataset | 44,000 real articles | ~98–99% |

For the real dataset, download from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` + `True.csv` in the project folder, then update `fake_news_detector.py`:
```python
# df = generate_sample_data(n=500)   ← comment out
df = load_real_data("Fake.csv", "True.csv")  # ← uncomment
```

---

## 🤖 Models Compared

| Model | Speed | Accuracy | Best For |
|-------|-------|----------|----------|
| Logistic Regression | Fast | ~98% | Best overall |
| Naive Bayes | Fastest | ~95% | Small datasets |
| Random Forest | Slow | ~97% | Noisy data |

---

## 🧠 How It Works
```
Input Text → Clean → TF-IDF Vectorize → Train 3 Models → Pick Best → Predict
```

The website sends text to the **Claude AI API** which returns verdict, probabilities, confidence level, reasoning, and signal tags as JSON.

---

## ⚙️ Requirements
```
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
joblib>=1.3.0
```

---

## 🛠️ Troubleshooting

| Error | Fix |
|-------|-----|
| `No module named 'pandas'` | `pip install pandas numpy scikit-learn joblib` |
| `'pip' is not recognized | `python -m pip install -r requirements.txt` |
| Path not found | Wrap folder path in quotes: `cd "C:\path\to\fake news detector"` |
| Website no result | Check internet — offline fallback activates automatically |

---

## 🔧 Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.9+ · HTML/CSS/JS |
| ML | Scikit-learn (TF-IDF, LR, NB, RF) |
| AI API | Anthropic Claude API |
| Data | Pandas, NumPy |
| Persistence | Joblib |

---

*Built by Kamali R · TruthScan Fake News Detector*
