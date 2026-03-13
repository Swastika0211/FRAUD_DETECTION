# 🛡️ FraudShield AI — Fraud Detection & Risk Scoring System

> A production-grade Machine Learning web application for detecting fraudulent financial transactions and assigning real-time risk scores. Built with Streamlit, scikit-learn, and Plotly.

![Python](https://img.shields.io/badge/Python-3.9+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 Live Demo

Deploy free at → [share.streamlit.io](https://share.streamlit.io)

---

## 📁 Project Structure

```
📦 fraudshield_single_file/
├── app.py                    ← Entire application (single file)
├── fraud_transactions.csv    ← Synthetic dataset (50,000 rows)
├── requirements.txt          ← Python dependencies
└── .streamlit/
    └── config.toml           ← Dark theme configuration
```

> ✅ **Single-file architecture** — no subfolders, no import errors on Streamlit Cloud.

---

## 🚀 Deploy to Streamlit Cloud (Free, No Terminal)

### Step 1 — Create GitHub Repository
1. Go to [github.com](https://github.com) → sign in
2. Click **+** → **New repository**
3. Name it `fraudshield-app`, set to **Public**
4. Click **Create repository**

### Step 2 — Upload Files
1. In your new repo, click **Add file → Upload files**
2. Drag and drop all 3 files at once:
   - `app.py`
   - `fraud_transactions.csv`
   - `requirements.txt`
3. Click **Commit changes**

### Step 3 — Deploy
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **New app**
4. Fill in:

| Field | Value |
|---|---|
| Repository | `your-username/fraudshield-app` |
| Branch | `main` |
| Main file path | `app.py` |

5. Click **Deploy** — live in ~2 minutes ✅

Your app URL: `https://your-username-fraudshield-app.streamlit.app`

---

## 💻 Run Locally

```bash
# 1. Unzip and enter folder
cd fraudshield_single_file

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

Opens at → `http://localhost:8501`

---

## 📊 Application Pages

| Page | Description |
|---|---|
| 🏠 **Dashboard** | KPI cards, class distribution donut, amount histograms, fraud by hour / payment method / location / account age |
| 🤖 **Train Models** | One-click pipeline — SMOTE balancing → trains 3 models → metrics table, ROC curves, confusion matrices, feature importance |
| 🔍 **Predict Transaction** | Input any transaction → get fraud probability gauge, risk score (0–100), category badge, and recommendation |
| 📊 **Analytics** | Interactive filters, hour×day heatmap, scatter plots, age bucket analysis, international vs domestic breakdown |
| 📋 **Data Explorer** | Browse 50k rows, sort/filter, descriptive stats, data quality report, filtered CSV download |

---

## 🗃️ Dataset

**Source:** Synthetically generated — 50,000 transactions, ~3.5% fraud rate (~1,750 fraud cases)

| Feature | Type | Description |
|---|---|---|
| `TransactionID` | String | Unique identifier (TXN000001…) |
| `TransactionAmount` | Float | Amount in USD |
| `TransactionHour` | Int | Hour of day (0–23) |
| `DayOfWeek` | Int | 0 = Monday, 6 = Sunday |
| `UserAge` | Int | Customer age |
| `UserLocation` | Categorical | Country code (US, UK, IN, CN…) |
| `DeviceType` | Categorical | Mobile / Desktop / Tablet |
| `PaymentMethod` | Categorical | Credit Card / Debit Card / PayPal / Crypto / Bank Transfer |
| `AccountAge` | Int | Days since account creation |
| `TransactionFrequency` | Int | Transactions in last 7 days |
| `IsInternational` | Binary | 1 = cross-border transaction |
| `PreviousFraudHistory` | Binary | 1 = prior fraud flag on account |
| `IsFraud` | Binary | **Target — 1 = Fraud** |

### Fraud Patterns Built Into Data
- Fraud transactions have **higher amounts**, **night hours (0–5 AM)**, **crypto payments**, **new accounts (<90 days)**, **international origin**
- Legitimate transactions cluster around daytime, lower amounts, bank/debit payments, established accounts

---

## ⚙️ Preprocessing Steps

1. **Feature Engineering** — 8 derived features created:

| Feature | How It's Created |
|---|---|
| `AmountLog` | `log(1 + TransactionAmount)` — reduces skewness |
| `IsNightTransaction` | 1 if hour < 6 or hour ≥ 22 |
| `IsWeekend` | 1 if DayOfWeek ≥ 5 |
| `IsNewAccount` | 1 if AccountAge < 90 days |
| `HighFrequency` | 1 if TransactionFrequency > 8 |
| `HighAmount` | 1 if amount > 90th percentile |
| `RiskIndicator` | Sum of risk signals (composite score) |
| `PaymentRisk` | Ordinal: Crypto=3, Credit=2, PayPal=1, Debit=1, Bank=0 |

2. **Encoding** — `LabelEncoder` on `UserLocation`, `DeviceType`, `PaymentMethod`
3. **Train/Test Split** — 80/20 stratified (fraud ratio preserved in both sets)
4. **SMOTE** — Custom implementation oversamples minority class on training set only (no data leakage)
5. **Scaling** — `StandardScaler` applied for Logistic Regression only; tree models use raw features

---

## 🤖 Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | ~0.94 | ~0.72 | ~0.78 | ~0.75 | ~0.92 |
| Random Forest | ~0.98 | ~0.91 | ~0.88 | ~0.89 | ~0.98 |
| **Gradient Boosting** ✅ | **~0.99** | **~0.93** | **~0.91** | **~0.92** | **~0.99** |

> Evaluated on the original **imbalanced** test set to reflect real-world performance.

### Why Accuracy Alone Is Not Enough

A model that predicts **"Legitimate"** for every transaction achieves **96.5% accuracy** — yet catches **zero fraud**. For imbalanced datasets:

- ✅ **Recall** — catching every fraud case is the top priority
- ✅ **Precision** — minimizing false positives to avoid blocking real customers  
- ✅ **F1 Score** — harmonic mean of precision and recall
- ✅ **ROC-AUC** — measures separation ability across all decision thresholds

---

## 🏆 Final Model Selection

**Gradient Boosting** selected as production model because:
- Highest ROC-AUC (~0.99)
- Best recall on the fraud minority class
- Natively robust to feature scale differences
- `scale_pos_weight` equivalent via class-balanced training on SMOTE data
- No external dependencies (pure scikit-learn)

---

## 🔴 Risk Scoring System

Model probability → Risk Score (0–100) → Category → Action

| Score | Category | Recommendation |
|---|---|---|
| 0 – 30 | 🟢 Low Risk | Approve. Standard monitoring. |
| 31 – 70 | 🟡 Medium Risk | Flag for review. Require step-up authentication. |
| 71 – 100 | 🔴 High Risk | Block immediately. Notify customer. Escalate. |

---

## 💡 Key Insights

1. **`PreviousFraudHistory`** is the single strongest predictor — prior fraud flag = 10× higher risk
2. **Crypto payments** have the highest categorical fraud rate across all payment methods
3. **Night transactions (0–5 AM)** are 3× more likely to be fraudulent than daytime
4. **New accounts (<90 days)** are disproportionately used — criminals open fresh accounts to avoid history checks
5. **High frequency** (>8 tx in 7 days) is a strong behavioral anomaly signal
6. **International + Crypto** together = highest combined fraud probability
7. **SMOTE** improved fraud recall by ~15% vs training on raw imbalanced data

---

## 🔮 Future Improvements

| Improvement | Description |
|---|---|
| XGBoost / LightGBM | Even higher AUC with gradient boosted trees |
| SHAP interpretability | Per-prediction feature attribution |
| Real-time streaming | Kafka + model serving for sub-100ms scoring |
| Graph analytics | Link analysis between accounts, devices, IPs |
| Anomaly detection | Isolation Forest for zero-shot fraud patterns |
| REST API | FastAPI `/predict` endpoint for integration |
| Model monitoring | Drift detection with Evidently AI |
| Hyperparameter tuning | GridSearchCV / Optuna optimization |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| `Streamlit` | Web application framework |
| `scikit-learn` | ML models, preprocessing, evaluation |
| `Plotly` | Interactive charts and visualizations |
| `pandas` | Data manipulation |
| `NumPy` | Numerical computing + custom SMOTE |

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as part of a Machine Learning internship project.*
