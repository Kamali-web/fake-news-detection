"""
========================================
  FAKE NEWS DETECTION SYSTEM
  Using NLP + Scikit-learn
========================================
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.pipeline import Pipeline
import joblib


# ──────────────────────────────────────────────
# 1. SAMPLE DATA GENERATOR (no download needed)
# ──────────────────────────────────────────────

def generate_sample_data(n=500):
    """
    Creates a small synthetic dataset so you can run this immediately.
    Replace this with a real CSV (see load_real_data()) for production use.
    """
    real_news = [
        "Scientists discover new vaccine effective against multiple flu strains",
        "Federal Reserve raises interest rates by 0.25 percent",
        "NASA successfully launches Mars rover mission",
        "Stock markets rise after positive jobs report",
        "Government passes new infrastructure spending bill",
        "University researchers publish study on climate change effects",
        "Tech company announces quarterly earnings beat expectations",
        "Olympic athletes prepare for upcoming games",
        "New study links exercise to improved mental health outcomes",
        "City council approves plan for new public transportation system",
        "Scientists find evidence of water on distant moon",
        "Election commission certifies results after recount",
        "Hospital introduces new robotic surgery technique",
        "International trade agreement signed between nations",
        "Central bank releases annual economic forecast report",
    ]
    fake_news = [
        "Drinking bleach cures all diseases according to secret doctors",
        "Aliens have landed and government is covering it up",
        "Bill Gates microchipped everyone through water supply last year",
        "The moon is actually a hologram created by elites",
        "Eating raw garlic daily makes you completely immune to cancer",
        "Secret underground tunnels connect major world cities for elites",
        "5G towers are being used to control human thoughts",
        "The president was secretly replaced by a robot last month",
        "Ancient Egyptians had smartphones hidden by mainstream historians",
        "Drinking urine can cure every known illness according to whistleblowers",
        "Sharks have gone extinct but media hides this truth from you",
        "The earth is expanding and scientists are paid to lie about it",
        "Vaccines contain nanobots that spy on you for corporations",
        "Flat earth proof found in classified NASA documents leaked today",
        "Celebrities are being replaced by clones in underground labs",
    ]

    np.random.seed(42)
    texts, labels = [], []

    for _ in range(n // 2):
        base = np.random.choice(real_news)
        # add slight variation
        noise = np.random.choice(["", " experts say", " reports show", " data confirms"])
        texts.append(base + noise)
        labels.append(1)  # REAL

    for _ in range(n // 2):
        base = np.random.choice(fake_news)
        noise = np.random.choice(["", " shocking truth", " they hide this", " exposed"])
        texts.append(base + noise)
        labels.append(0)  # FAKE

    df = pd.DataFrame({"text": texts, "label": labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


# ──────────────────────────────────────────────
# 2. LOAD REAL DATA (optional – Kaggle dataset)
# ──────────────────────────────────────────────

def load_real_data(fake_csv="Fake.csv", true_csv="True.csv"):
    """
    Download from Kaggle: 'clmentbisaillon/fake-and-real-news-dataset'
    Place Fake.csv and True.csv in the same folder as this script.
    """
    fake = pd.read_csv(fake_csv)
    real = pd.read_csv(true_csv)
    fake["label"] = 0
    real["label"] = 1
    df = pd.concat([fake, real], ignore_index=True)
    df["text"] = df["title"].fillna("") + " " + df["text"].fillna("")
    return df[["text", "label"]].sample(frac=1, random_state=42).reset_index(drop=True)


# ──────────────────────────────────────────────
# 3. TEXT PREPROCESSING
# ──────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                    # remove HTML tags
    text = re.sub(r"\[.*?\]", "", text)                  # remove square brackets
    text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)  # punctuation
    text = re.sub(r"\d+", "", text)                      # remove numbers
    text = re.sub(r"\s+", " ", text).strip()             # extra whitespace
    return text


# ──────────────────────────────────────────────
# 4. BUILD MODELS
# ──────────────────────────────────────────────

def build_pipelines():
    tfidf = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),       # unigrams + bigrams
        stop_words="english",
        sublinear_tf=True,
    )

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf),
            ("clf", LogisticRegression(max_iter=1000, C=1.0)),
        ]),
        "Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=10_000, ngram_range=(1, 2),
                                      stop_words="english")),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5_000, ngram_range=(1, 1),
                                      stop_words="english")),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ]),
    }
    return models


# ──────────────────────────────────────────────
# 5. TRAIN & EVALUATE
# ──────────────────────────────────────────────

def train_and_evaluate(df):
    print("\n📰  FAKE NEWS DETECTION SYSTEM")
    print("=" * 50)

    # Preprocess
    print("\n🔧  Preprocessing text...")
    df["clean_text"] = df["text"].apply(clean_text)
    print(f"    Dataset size : {len(df):,} articles")
    print(f"    Real news    : {df['label'].sum():,} ({df['label'].mean()*100:.1f}%)")
    print(f"    Fake news    : {(df['label']==0).sum():,} ({(1-df['label'].mean())*100:.1f}%)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["label"],
        test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\n📊  Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Train all models
    models = build_pipelines()
    results = {}
    best_model, best_acc = None, 0

    print("\n🤖  Training models...\n")
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc   = accuracy_score(y_test, y_pred)
        auc   = roc_auc_score(y_test, y_prob)
        results[name] = {"accuracy": acc, "roc_auc": auc, "pipeline": pipeline}

        print(f"  ✅  {name}")
        print(f"      Accuracy : {acc*100:.2f}%")
        print(f"      ROC-AUC  : {auc:.4f}")
        print()

        if acc > best_acc:
            best_acc   = acc
            best_model = (name, pipeline)

    # Detailed report for best model
    name, pipeline = best_model
    y_pred = pipeline.predict(X_test)

    print("=" * 50)
    print(f"🏆  BEST MODEL: {name}  ({best_acc*100:.2f}% accuracy)")
    print("=" * 50)
    print("\n📋  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

    print("🔢  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"         Predicted FAKE  Predicted REAL")
    print(f"  Actual FAKE    {cm[0][0]:>5}          {cm[0][1]:>5}")
    print(f"  Actual REAL    {cm[1][0]:>5}          {cm[1][1]:>5}")

    return pipeline, results


# ──────────────────────────────────────────────
# 6. PREDICT SINGLE ARTICLE
# ──────────────────────────────────────────────

def predict_article(pipeline, text: str) -> dict:
    cleaned = clean_text(text)
    pred    = pipeline.predict([cleaned])[0]
    proba   = pipeline.predict_proba([cleaned])[0]

    return {
        "label"      : "✅ REAL" if pred == 1 else "❌ FAKE",
        "confidence" : f"{max(proba)*100:.1f}%",
        "real_prob"  : f"{proba[1]*100:.1f}%",
        "fake_prob"  : f"{proba[0]*100:.1f}%",
    }


# ──────────────────────────────────────────────
# 7. SAVE & LOAD MODEL
# ──────────────────────────────────────────────

def save_model(pipeline, path="fake_news_model.pkl"):
    joblib.dump(pipeline, path)
    print(f"\n💾  Model saved → {path}")

def load_model(path="fake_news_model.pkl"):
    return joblib.load(path)


# ──────────────────────────────────────────────
# 8. MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":

    # ── Load data ──────────────────────────────
    # Option A: synthetic data (works immediately, no downloads)
    df = generate_sample_data(n=500)

    # Option B: real Kaggle dataset (uncomment after downloading CSVs)
    # df = load_real_data("Fake.csv", "True.csv")

    # ── Train & evaluate ───────────────────────
    best_pipeline, all_results = train_and_evaluate(df)

    # ── Save model ─────────────────────────────
    save_model(best_pipeline)

    # ── Live predictions ───────────────────────
    print("\n🔍  LIVE PREDICTIONS")
    print("=" * 50)

    test_articles = [
        "Researchers at MIT published a peer-reviewed study showing new solar panels achieve record efficiency",
        "Secret chemtrails are poisoning the water supply and the government is covering it up",
        "The central bank announced a quarter-point rate cut following last month's inflation report",
        "Drinking hot lemon water every morning cures cancer according to hidden medical documents",
    ]

    for article in test_articles:
        result = predict_article(best_pipeline, article)
        print(f"\n  📰  {article[:70]}...")
        print(f"      → {result['label']}  (confidence: {result['confidence']})")
        print(f"         Real: {result['real_prob']}  |  Fake: {result['fake_prob']}")

    # ── Interactive mode ───────────────────────
    print("\n\n💬  INTERACTIVE MODE  (type 'quit' to exit)")
    print("=" * 50)
    while True:
        user_input = input("\nPaste a news headline or article:\n> ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye! 👋")
            break
        if not user_input:
            continue
        result = predict_article(best_pipeline, user_input)
        print(f"\n  Result     : {result['label']}")
        print(f"  Confidence : {result['confidence']}")
        print(f"  Real prob  : {result['real_prob']}")
        print(f"  Fake prob  : {result['fake_prob']}")