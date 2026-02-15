# � F1 Winner & Podium Prediction using Machine Learning

## 📌 Project Overview

This project builds advanced Machine Learning models to **predict Formula 1 podium finishes (Top 3)** using comprehensive historical race data from 2018-2024.

The models analyze driver performance, qualifying position, grid advantage, and constructor statistics to forecast which drivers will finish in the top 3 positions at any given race, with **99.33% test accuracy!**

### 🎯 Key Features:
- ✅ **Logistic Regression** - 99.33% accuracy (Best Model)
- ✅ **Random Forest** - 97.15% accuracy
- ✅ **2025 Season Predictions** - Based on actual 2024 data
- ✅ **9 Engineered Features** - Advanced performance metrics
- ✅ **2,976 Race Records** - Complete 2018-2024 dataset

---

## 📊 Dataset Source

**Local CSV Files (2018-2024 F1 Seasons)**
- `races.csv` - Race schedule and details
- `results.csv` - Final race results
- `drivers.csv` - Driver information
- `constructors.csv` - Team information
- `qualifying.csv` - Qualifying session results
- `driver_standings.csv` - Championship standings
- Plus 8 additional datasets for comprehensive analysis

**Note:** Previously used Ergast F1 API (now shut down). Project now uses pre-downloaded CSV datasets for reliability.

---

## ⚙️ Features Engineered

| Feature | Description | Importance |
|---------|-------------|-----------|
| **GridImprovement** | Grid position - Finish position (overtaking ability) | 20.6% |
| **QualifyingPosition** | Starting grid position from qualifying | 20.2% |
| **CurrentSeasonPointsPerRace** | Driver's season points average | 17.0% |
| **Last5FinishAvg** | Average finish position in last 5 races | 15.8% |
| **GridPosition** | Starting grid placement | 10.4% |
| **ConstructorAvgPosition** | Team average finish position | 7.5% |
| **WinRate** | Career win percentage | 4.0% |
| **CareerRaces** | Total races competed | 2.9% |
| **CareerWin** | Total career wins | 1.6% |

---

## 🧠 Models & Performance

### 1️⃣ Logistic Regression (BEST MODEL ⭐)
```
Training Accuracy:  99.50%
Test Accuracy:      99.33%
Precision (Top 3):  100%
Recall (Top 3):     96%
```
**Best for:** Production predictions, interpretability, fast inference

### 2️⃣ Random Forest
```
Training Accuracy:  99.96%
Test Accuracy:      97.15%
Precision (Top 3):  93%
Recall (Top 3):     89%
```
**Best for:** Feature importance analysis, catching non-linear patterns

---

## 📈 Model Evaluation Outputs

The project generates comprehensive analysis:
- ✅ Accuracy Score (both models)
- ✅ Classification Reports
- ✅ Confusion Matrices
- ✅ Feature Importance Rankings
- ✅ Performance Visualizations
- ✅ 2024 Season Analysis
- ✅ 2025 Season Predictions

---

## 📂 Project Structure

```
F1-Winner-Prediction/
│
├── F1.py                          # Main prediction script
├── f1_winner_prediction.py        # Alternative entry point
├── README.md                       # This file
│
├── Data Files (CSV):
│   ├── races.csv
│   ├── results.csv
│   ├── drivers.csv
│   ├── constructors.csv
│   ├── qualifying.csv
│   ├── driver_standings.csv
│   ├── constructor_standings.csv
│   ├── pit_stops.csv
│   ├── lap_times.csv
│   ├── seasons.csv
│   ├── sprint_results.csv
│   └── status.csv
│
├── Visualizations (Generated):
│   ├── f1_analysis.png            # Data distribution analysis
│   ├── model_comparison.png       # Model performance comparison
│   └── confusion_matrices.png     # Prediction accuracy matrices
│
└── .venv/                         # Python virtual environment
```

---

## 🚀 Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Suniltkumartsk/Formula-1-F1.git
cd Formula-1-F1
```

### 2️⃣ Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 4️⃣ Run the Project
```bash
python F1.py
```

Or use the virtual environment Python:
```bash
.venv\Scripts\python.exe F1.py    # Windows
.venv/bin/python F1.py             # macOS/Linux
```

---

## 📦 Requirements

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

---

## ▶️ What the Script Does

1. **Loads Data** - Reads all 13 CSV files for comprehensive F1 datasets
2. **Data Processing** - Filters 2018-2024 seasons, merges datasets
3. **Feature Engineering** - Creates 9 advanced performance metrics
4. **Model Training** - Trains Logistic Regression & Random Forest on 2,380 samples
5. **Evaluation** - Tests on 596 samples with detailed metrics
6. **Visualization** - Generates 3 high-quality analysis charts
7. **2024 Analysis** - Shows podium contenders for 2024 season
8. **2025 Predictions** - Predicts likely podium finishes for 2025

---

## 🔍 2025 Season Predictions (Top Contenders)

Based on actual 2024 season data:

| Rank | Driver | Grid Avg | Quali Avg | Top 3 Probability |
|------|--------|----------|-----------|------------------|
| 1 | **Max Verstappen** ⭐ | 3.5 | 2.9 | **72.01%** |
| 2 | **Lando Norris** | 3.4 | 3.5 | **32.81%** |
| 3 | **Charles Leclerc** | 5.4 | 5.3 | **23.57%** |
| 4 | Oscar Piastri | 5.4 | 5.4 | 8.94% |
| 5 | Carlos Sainz | 4.8 | 5.5 | 4.31% |

---

## 🎯 Key Insights from Analysis

- 🏆 **Qualifying position is crucial** - Most important predictor (20.2% importance)
- 🚀 **Grid improvement matters** - Overtaking ability strongly influences podium (20.6%)
- 📈 **Recent form is telling** - Last 5 races average predicts 15.8% of outcomes
- 🔴 **Grid position advantage** - Starting in top positions increases podium chance significantly
- 🏁 **Consistency wins races** - Season points per race is highly predictive

---

## 📊 Example Model Output

```
Loading data from CSV files...
Total races in dataset (2018-2024): 149
Total Recorded Races: 2979
Total Record After cleaning: 2976

EXPLORATORY DATA ANALYSIS
Dataset Overview: 2,976 records with 18 columns

MODEL PREPARATION & TRAINING
Training set size: 2,380
Test set size: 596

LOGISTIC REGRESSION MODEL
Training Accuracy: 0.9950
Test Accuracy: 0.9933 ✓

RANDOM FOREST MODEL
Training Accuracy: 0.9996
Test Accuracy: 0.9715

2025 SEASON PREDICTIONS (UPCOMING)
Max Verstappen - Grid: 3.5, Quali: 2.9 - Top 3 Prob: 72.01%
```

---

## 💼 Use Cases

- 📊 **Sports Analytics** - Predict race outcomes for fantasy F1 leagues
- 💰 **Betting Predictions** - Inform podium finish bets
- 📰 **Journalism** - Pre-race analysis and predictions
- 🎓 **Education** - Learn ML pipeline on real sports data
- 🏎️ **Team Strategy** - Analyze driver performance trends

---

## 🚀 Future Enhancements

- [ ] Add weather condition data
- [ ] Include track-specific performance history
- [ ] Implement XGBoost/LightGBM for higher accuracy
- [ ] Deploy as Streamlit web application
- [ ] Add real-time 2025 season updates
- [ ] Create driver comparison tool
- [ ] Add constructors championship predictions
- [ ] Integration with live race data APIs

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ **Data Loading & Merge** - Working with multiple CSV datasets
- ✅ **Feature Engineering** - Creating meaningful predictive features
- ✅ **Model Comparison** - Evaluating different ML algorithms
- ✅ **Performance Metrics** - Accuracy, Precision, Recall, Confusion Matrix
- ✅ **Data Visualization** - Creating insightful charts
- ✅ **Time-Series Analysis** - Performance trends over seasons
- ✅ **Production-Ready Code** - Well-structured, documented pipeline

---

## 👥 Perfect For:

- 🎯 Data Science Portfolio Projects
- 📚 Machine Learning Portfolio
- 🏆 Interview Preparation (FAANG)
- 👔 Sports Analytics Positions
- 🤖 ML Engineer Internships

---

## 📜 License

Open-source for educational and personal use.

---

## 🤝 Contributing

Contributions welcome! Feel free to:
- Submit bug reports
- Add new features
- Improve documentation
- Share prediction insights

---

## 📧 Contact & Repository

**GitHub:** [github.com/Suniltkumartsk/Formula-1-F1](https://github.com/Suniltkumartsk/Formula-1-F1)

**Project Status:** ✅ Complete & Deployed

---

**Last Updated:** February 15, 2026
**Python Version:** 3.11.9
**Status:** Fully Functional ✓

