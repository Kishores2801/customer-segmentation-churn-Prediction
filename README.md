# 📉 Customer Segmentation & Churn Prediction — Telecom Industry

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![GCP](https://img.shields.io/badge/Google_Cloud_Run-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)](https://cloud.google.com/run)

> **An end-to-end Machine Learning application that predicts customer churn and performs customer segmentation for the telecom industry — with a live interactive Gradio interface deployed on Google Cloud Run.**

🌐 **Live Demo:** [https://churn-predictor-service-285117395810.us-central1.run.app/](https://churn-predictor-service-285117395810.us-central1.run.app/)

> ⚠️ *Note: This repository was re-initialized due to a corrupted local git history. The project itself was developed and iterated over several months.*

---

## 📋 Table of Contents

- [Overview](#overview)
- [Business Problem](#business-problem)
- [Features](#features)
- [Key Features Used for Prediction](#key-features-used-for-prediction)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Customer Segmentation](#customer-segmentation)
- [Churn Prediction Models](#churn-prediction-models)
- [Getting Started (Local)](#getting-started-local)
- [Docker Setup](#docker-setup)
- [Deployment on Google Cloud Run](#deployment-on-google-cloud-run)
- [Live App Usage](#live-app-usage)
- [License](#license)

---

## 🧭 Overview

This project tackles one of the most impactful problems in the telecom industry: **customer churn**. By combining **unsupervised segmentation** (clustering customers into meaningful groups) with **supervised churn prediction** (classifying which customers are likely to leave), this project delivers both strategic insights and actionable predictions.

The solution is fully productionized — trained models are served through a **Gradio web interface**, containerized with **Docker**, and deployed on **Google Cloud Run** for scalable, serverless access.

---

## 💼 Business Problem

Telecom companies lose billions annually to customer churn. Acquiring a new customer costs **5–10x more** than retaining an existing one. The ability to:

1. **Identify which customers are at risk** of churning before they leave
2. **Understand customer segments** to personalize retention strategies

...directly translates to revenue saved and customer lifetime value increased.

This project answers both questions with data-driven ML models.

---

## ✨ Features

- 🔍 **Customer Segmentation** — Unsupervised clustering to group customers by behavior and usage patterns
- 🤖 **Churn Prediction** — Multiple ML models trained and compared to predict churn probability
- 🧠 **Deep Learning Model** — Neural network built with TensorFlow/Keras for churn classification
- 📊 **Exploratory Data Analysis** — In-depth EDA notebooks with visualizations using Plotly and Seaborn
- 🖥️ **Interactive Gradio App** — Live web interface to input customer data and get real-time predictions
- 🐳 **Dockerized** — Fully containerized for reproducible local and cloud deployment
- ☁️ **Google Cloud Run** — Deployed as a serverless container with public access

---

## 🔑 Key Features Used for Prediction

The model was trained on **5 carefully selected features**, all of which are observable *before* a customer churns — ensuring zero data leakage and full real-world applicability.

| Feature | Business Meaning | Churn Signal |
|---|---|---|
| **Total Spend** | Customer lifetime monetary value | High/low spend profiles churn differently |
| **Usage Frequency** | Product engagement & activity level | Declining usage = early disengagement warning |
| **Payment Delay** | Billing behaviour & financial reliability | Delays strongly correlate with dissatisfaction |
| **Support Calls** | Volume of customer service interactions | High call frequency = frustration indicator |
| **Last Interaction** | Recency of customer engagement | Long gap since last interaction = passive churn risk |

> 💡 These features follow the well-established **RFM (Recency, Frequency, Monetary)** framework used in customer analytics, extended with service experience signals. All features are **pre-churn observable** — meaning there is no target leakage, and the model can be applied to active customers in real-time.

---

## 📁 Project Structure

```
customer-segmentation-churn-Prediction/
├── notebooks/                  # Jupyter Notebooks for EDA, segmentation & model training
│   ├── 01_eda.ipynb            # Exploratory Data Analysis
│   ├── 02_segmentation.ipynb   # Customer Segmentation (Clustering)
│   └── 03_churn_prediction.ipynb # Model Training & Evaluation
├── models/                     # Saved trained models (.pkl, .h5, .keras)
├── src/                        # Modular Python source code
│   ├── preprocessing.py        # Data cleaning & feature engineering
│   ├── segmentation.py         # Clustering logic
│   ├── prediction.py           # Churn prediction inference
│   └── utils.py                # Helper functions
├── .vscode/                    # VS Code workspace settings
├── Dockerfile                  # Docker container configuration
├── .dockerignore               # Files excluded from Docker build
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python dependencies
├── main.py                     # Gradio app entry point
└── README.md
```

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.11 |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn |
| **Deep Learning** | TensorFlow, Keras |
| **Visualization** | Plotly, Matplotlib, Seaborn |
| **Web Interface** | Gradio |
| **Data Format** | PyArrow (Parquet support) |
| **Containerization** | Docker (python:3.11-slim) |
| **Cloud Hosting** | Google Cloud Run |

---

## 🔬 ML Pipeline

```
Raw Telecom Data
       │
       ▼
┌─────────────────┐
│  Data Cleaning  │  → Handle missing values, outliers, type casting
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│           Feature Selection              │
│  Total Spend · Usage Frequency           │
│  Payment Delay · Support Calls           │
│  Last Interaction                        │
└────────┬─────────────────────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐  ┌──────────────┐
│Segment │  │Churn         │
│Analysis│  │Prediction    │
│(K-Means│  │(ANN / SKLearn│
│Cluster)│  │  Models)     │
└────────┘  └──────────────┘
    │               │
    ▼               ▼
Customer        Churn Risk
Profiles        Score (0–1)
(Segments)      F1: 0.93
    │               │
    └───────┬───────┘
            ▼
     Gradio Web App
     (Live on GCP ☁️)
```

---

## 👥 Customer Segmentation

Customers are grouped into distinct segments using **unsupervised clustering** techniques based on the same 5 features used for prediction:

| Segment Driver | What It Reveals |
|---|---|
| **Total Spend** | Separates high-value vs low-value customers |
| **Usage Frequency** | Identifies highly engaged vs dormant users |
| **Payment Delay** | Flags financially stressed or unreliable payers |
| **Support Calls** | Groups frustrated vs satisfied customers |
| **Last Interaction** | Distinguishes active vs disengaged customers |

Each resulting cluster is profiled to understand its behavioral fingerprint — enabling targeted, segment-specific retention campaigns (e.g. loyalty discounts for high-value at-risk customers, re-engagement offers for dormant users).

---

## 🤖 Churn Prediction Models

Multiple models were trained and evaluated for churn classification:

| Model | Type | Notes |
|---|---|---|
| Logistic Regression | Baseline | Interpretable, fast |
| Random Forest | Ensemble | Feature importance analysis |
| Gradient Boosting | Ensemble | Strong performance on tabular data |
| **Neural Network (ANN)** ⭐ | **Deep Learning** | **Best model — deployed to production** |

---

### 🏆 Best Model — TensorFlow ANN Results

Evaluated on **88,167 test samples**:

| Metric | Class 0 (No Churn) | Class 1 (Churn) | Overall |
|---|---|---|---|
| **Precision** | 0.86 | 0.99 | — |
| **Recall** | 0.99 | 0.88 | — |
| **F1-Score** | 0.92 | **0.93** | — |
| **Accuracy** | — | — | **93%** |
| **Macro Avg F1** | — | — | **0.93** |
| **Weighted Avg F1** | — | — | **0.93** |

```
              precision    recall  f1-score   support

         0.0       0.86      0.99      0.92     38,063
         1.0       0.99      0.88      0.93     50,104

    accuracy                           0.93     88,167
   macro avg       0.93      0.93      0.93     88,167
weighted avg       0.94      0.93      0.93     88,167
```

> 🔑 **Key Insight:** The model achieves **99% precision on churn class (1.0)** — meaning when it predicts a customer will churn, it is almost always correct. This is critical for telecom businesses to confidently act on predictions without wasting retention resources on false positives.

**Evaluation Metrics used:**
- Accuracy
- Precision & Recall per class
- F1 Score (macro & weighted)
- Confusion Matrix
- ROC-AUC Curve

> The best performing ANN model is saved to the `models/` directory and served via the Gradio app.

---

## 🚀 Getting Started (Local)

### 1. Clone the Repository

```bash
git clone https://github.com/Kishores2801/customer-segmentation-churn-Prediction.git
cd customer-segmentation-churn-Prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Gradio App

```bash
python main.py
```

Open **[http://localhost:8080](http://localhost:8080)** in your browser.

### 5. Run the Notebooks (Optional)

```bash
pip install jupyter
jupyter notebook notebooks/
```

---

## 🐳 Docker Setup

Build and run the app locally using Docker:

```bash
# Build the image
docker build -t churn-predictor .

# Run the container
docker run -p 8080:8080 churn-predictor
```

Open **[http://localhost:8080](http://localhost:8080)** in your browser.

---

## ☁️ Deployment on Google Cloud Run

This project is deployed on **Google Cloud Run** as a containerized Gradio application.

### Step 1 — Build & Push the Docker Image

```bash
docker build -t gcr.io/<YOUR_PROJECT_ID>/churn-predictor-service .
docker push gcr.io/<YOUR_PROJECT_ID>/churn-predictor-service
```

### Step 2 — Deploy to Cloud Run

```bash
gcloud run deploy churn-predictor-service \
  --image gcr.io/<YOUR_PROJECT_ID>/churn-predictor-service \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080
```

> ✅ Cloud Run will return a public URL once deployment is complete.

---

## 🖥️ Live App Usage

Visit the live app at:
**[https://churn-predictor-service-285117395810.us-central1.run.app/](https://churn-predictor-service-285117395810.us-central1.run.app/)**

The Gradio interface allows you to:

1. **Input 5 customer attributes:**
   - `Total Spend` — customer's total historical spend
   - `Usage Frequency` — how frequently they use the service
   - `Payment Delay` — average number of days payments are delayed
   - `Support Calls` — number of support calls made
   - `Last Interaction` — days since last interaction with the service
2. **Get churn prediction** — binary classification (Churn / No Churn) with probability score
3. **View customer segment** — which behavioural cluster the customer belongs to
4. **Interpret results** — understand the key factors driving the prediction

---

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow
keras
jupyter
gradio
plotly
pyarrow
```

Install all with:
```bash
pip install -r requirements.txt
```

---

## 📄 License

This project is open source and available under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

Built with 🐍 Python · 🤖 TensorFlow · 🖥️ Gradio · ☁️ Google Cloud Run

**[🌐 Try the Live App](https://churn-predictor-service-285117395810.us-central1.run.app/)**

</div>
