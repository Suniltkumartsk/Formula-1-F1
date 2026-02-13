# 🏎️ F1 Race Winner Prediction using Machine Learning
## 📌 Project Overview

This project builds a Machine Learning model to predict Formula 1 race winners using historical race data fetched from the Ergast F1 API.

# The model uses:

* Driver performance metrics
* Qualifying position
* Grid position
* Constructor performance
* Season points


 It compares:

✅ Logistic Regression

✅ Random Forest Classifier


#
#  🏁 Dataset Source

Data is fetched dynamically from:

## 🔗 Ergast F1 API
Seasons covered: 2018 – 2024

No manual dataset download required.

## ⚙️ Features Engineered

The model includes advanced racing insights:

| Feature                    | Description                    |
| -------------------------- | ------------------------------ |
| GridPosition               | Starting grid position         |
| QualifyingPosition         | Qualifying session rank        |
| PreRacePoints              | Driver points before race      |
| Last5RacesAvgPosition      | Driver form indicator          |
| WinRate                    | Career win percentage          |
| GridImprovement            | Position gain/loss             |
| CurrentSeasonPointsPerRace | Season performance consistency |



## Target Variable:

Won (1 = Winner, 0 = Not Winner)

# 🧠 Models Used
## 1️⃣ Logistic Regression

* Interpretable
* Baseline model
* Good for linear relationships

-----
## 2️⃣ Random Forest

* Captures non-linear patterns
* Handles feature interactions
* Generally higher accuracy
-----

# 📊 Model Evaluation

## The project evaluates:

* Accuracy Score

* Classification Report

* Confusion Matrix

* ROC-AUC Score

* Feature Importance Visualization

## It also generates:

📈 f1_winner_prediction_analysis.png
----

## 📂 Project Structure

    f1-winner-prediction/
    │
    ├── f1_winner_prediction.py
    ├── f1_winner_prediction_analysis.png
    ├── README.md
    └── requirements.txt

------------------------------------------------------------------------

# 🚀 Installation & Setup

## 1️⃣ Clone Repository

``` bash
git clone https://github.com/yourusername/f1-winner-prediction.git
cd f1-winner-prediction
```

------------------------------------------------------------------------

## 2️⃣ Create Virtual Environment (Recommended)

``` bash
python -m venv venv
venv\Scripts\activate   # Windows
```

------------------------------------------------------------------------

## 3️⃣ Install Dependencies

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📦 Requirements

    pandas
    numpy
    matplotlib
    seaborn
    scikit-learn
    requests

------------------------------------------------------------------------

# ▶️ Run the Project

``` bash
python f1_winner_prediction.py
```

------------------------------------------------------------------------

## 📌 What the Script Does

-   Fetch F1 historical data\
-   Perform feature engineering\
-   Train ML models\
-   Evaluate performance\
-   Generate visualizations\
-   Show example predictions

------------------------------------------------------------------------

# 🔍 Key Insights

-   🥇 Qualifying position is highly predictive of winning\
-   🏎️ Starting on the front row significantly increases win
    probability\
-   📈 Recent driver form strongly influences outcomes\
-   🌳 Random Forest outperforms Logistic Regression in capturing
    complex patterns

------------------------------------------------------------------------

# 📈 Future Improvements

-   Add weather conditions\
-   Include track-specific performance\
-   Use XGBoost / LightGBM\
-   Deploy as web app (Flask / Streamlit)\
-   Live prediction for upcoming races

------------------------------------------------------------------------

# 🎯 Resume Value

This project demonstrates:

-   API Data Collection\
-   Data Cleaning & Feature Engineering\
-   Time-based Train/Test Split\
-   Model Comparison\
-   Performance Evaluation\
-   Data Visualization\
-   End-to-End ML Pipeline

### 💼 Suitable For:

-   Data Analyst Roles\
-   ML Engineer Internships\
-   Sports Analytics Roles

------------------------------------------------------------------------

# 📜 License

Open-source for educational purposes.