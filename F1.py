import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings('ignore')

print("\nLoading data from CSV files...")

# Load all CSV files
races_df = pd.read_csv('races.csv')
results_df = pd.read_csv('results.csv')
qualifying_df = pd.read_csv('qualifying.csv')
drivers_df = pd.read_csv('drivers.csv')
constructors_df = pd.read_csv('constructors.csv')
driver_standings_df = pd.read_csv('driver_standings.csv')

# Extract year from date column in races
races_df['year'] = pd.to_datetime(races_df['date']).dt.year

# Filter for years 2018-2024
races_filtered = races_df[(races_df['year'] >= 2018) & (races_df['year'] <= 2024)].copy()

print(f"Total races in dataset (2018-2024): {len(races_filtered)}")

# Merge results with race information
data = results_df.merge(races_filtered[['raceId', 'year', 'round', 'name']], on='raceId', how='inner')

# Merge with drivers
data = data.merge(drivers_df[['driverId', 'surname', 'forename']], on='driverId', how='left')
data['DriverName'] = data['forename'] + ' ' + data['surname']

# Merge with constructors
data = data.merge(constructors_df[['constructorId', 'name']], on='constructorId', how='left', suffixes=('', '_constructor'))
data = data.rename(columns={'name_constructor': 'Constructor'})

# Merge with qualifying results
qualifying_df_clean = qualifying_df[['raceId', 'driverId', 'position']].copy()
qualifying_df_clean['position'] = pd.to_numeric(qualifying_df_clean['position'], errors='coerce')
qualifying_df_clean = qualifying_df_clean.rename(columns={'position': 'QualifyingPosition'})

data = data.merge(qualifying_df_clean, on=['raceId', 'driverId'], how='left')

# Create the final dataframe with required columns
df = pd.DataFrame({
    'Year': data['year'],
    'Round': data['round'],
    'RaceName': data['name'],
    'DriverID': data['driverId'],
    'DriverName': data['DriverName'],
    'Constructor': data['Constructor'],
    'FinishPosition': pd.to_numeric(data['positionOrder'], errors='coerce'),
    'GridPosition': pd.to_numeric(data['grid'], errors='coerce'),
    'QualifyingPosition': data['QualifyingPosition'],
    'RacePoints': pd.to_numeric(data['points'], errors='coerce').fillna(0),
})

print(f"Total Recorded Races: {len(df)}")

# Remove rows with missing critical values
df = df.dropna(subset=['FinishPosition', 'GridPosition', 'QualifyingPosition'])
print(f"Total Record After cleaning: {len(df)}")

# Feature Engineering

df = df.sort_values(['DriverID', 'Year', 'Round']).reset_index(drop=True)

# Ensure numeric types
df['RacePoints'] = pd.to_numeric(df['RacePoints'], errors='coerce').fillna(0)

df['Last5FinishAvg'] = df.groupby('DriverID')['FinishPosition'].transform(
    lambda x: x.rolling(window=5, min_periods=1).mean()
)

df['CareerRaces'] = df.groupby('DriverID').cumcount() + 1
df['CareerWin'] = df.groupby('DriverID')['FinishPosition'].transform(
    lambda x: (x == 1).cumsum()
)

df['WinRate'] = df['CareerWin'] / df['CareerRaces']
df['GridImprovement'] = df['GridPosition'] - df['FinishPosition']

# If they did not finish like DNF
df['Finished'] = 1

df = df.sort_values(['Year', 'Round']).reset_index(drop=True)

df['ConstructorAvgPosition'] = (
    df.groupby(['Year', 'Constructor'])['FinishPosition']
    .transform(lambda x: x.expanding().mean())
)

df['CurrentSeasonPointsPerRace'] = (
    df.groupby(['Year', 'DriverID'])['RacePoints'].cumsum() /
    (df.groupby(['Year', 'DriverID']).cumcount() + 1)
)

# ============= EXPLORATORY DATA ANALYSIS =============
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

print("\nDataset Overview:")
print(f"Total records: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic Statistics:\n{df.describe()}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('F1 Race Data Analysis (2018-2024)', fontsize=16)

# Points distribution
axes[0, 0].hist(df['RacePoints'], bins=30, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Distribution of Race Points')
axes[0, 0].set_xlabel('Points')
axes[0, 0].set_ylabel('Frequency')

# Finish Position distribution
axes[0, 1].hist(df['FinishPosition'], bins=20, color='lightcoral', edgecolor='black')
axes[0, 1].set_title('Distribution of Finish Positions')
axes[0, 1].set_xlabel('Position')
axes[0, 1].set_ylabel('Frequency')

# Grid vs Finish Position
axes[1, 0].scatter(df['GridPosition'], df['FinishPosition'], alpha=0.5)
axes[1, 0].set_title('Grid Position vs Finish Position')
axes[1, 0].set_xlabel('Grid Position')
axes[1, 0].set_ylabel('Finish Position')

# Win Rate by Constructor (top 10)
top_constructors = df['Constructor'].value_counts().head(10).index
win_rate_by_constructor = df[df['Constructor'].isin(top_constructors)].groupby('Constructor')['WinRate'].mean().sort_values(ascending=False)
axes[1, 1].barh(win_rate_by_constructor.index, win_rate_by_constructor.values, color='lightgreen')
axes[1, 1].set_title('Average Win Rate by Top Constructors')
axes[1, 1].set_xlabel('Win Rate')

plt.tight_layout()
plt.savefig('f1_analysis.png', dpi=100, bbox_inches='tight')
print("\n[+] Analysis plot saved as 'f1_analysis.png'")
plt.close()

# ============= MODEL PREPARATION =============
print("\n" + "="*50)
print("MODEL PREPARATION & TRAINING")
print("="*50)

# Create target variable: Did the driver finish in top 3?
df['FinishTop3'] = (df['FinishPosition'] <= 3).astype(int)

# Select features for modeling
feature_columns = [
    'GridPosition', 'QualifyingPosition', 'Last5FinishAvg',
    'CareerRaces', 'CareerWin', 'WinRate', 'GridImprovement',
    'ConstructorAvgPosition', 'CurrentSeasonPointsPerRace'
]

X = df[feature_columns].copy()
y = df['FinishTop3'].copy()

# Handle any remaining NaN values
X = X.fillna(X.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Target distribution (Training): \n{y_train.value_counts()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============= MODEL 1: LOGISTIC REGRESSION =============
print("\n" + "-"*50)
print("LOGISTIC REGRESSION MODEL")
print("-"*50)

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Predictions
lr_train_pred = lr_model.predict(X_train_scaled)
lr_test_pred = lr_model.predict(X_test_scaled)

# Evaluation
lr_train_acc = accuracy_score(y_train, lr_train_pred)
lr_test_acc = accuracy_score(y_test, lr_test_pred)

print(f"\nLogistic Regression Results:")
print(f"Training Accuracy: {lr_train_acc:.4f}")
print(f"Test Accuracy: {lr_test_acc:.4f}")
print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, lr_test_pred))

# ============= MODEL 2: RANDOM FOREST =============
print("-"*50)
print("RANDOM FOREST MODEL")
print("-"*50)

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf_model.fit(X_train_scaled, y_train)

# Predictions
rf_train_pred = rf_model.predict(X_train_scaled)
rf_test_pred = rf_model.predict(X_test_scaled)

# Evaluation
rf_train_acc = accuracy_score(y_train, rf_train_pred)
rf_test_acc = accuracy_score(y_test, rf_test_pred)

print(f"\nRandom Forest Results:")
print(f"Training Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy: {rf_test_acc:.4f}")
print(f"\nClassification Report (Test Set):")
print(classification_report(y_test, rf_test_pred))

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\nFeature Importance (Random Forest):")
print(feature_importance)

# ============= MODEL COMPARISON & VISUALIZATION =============
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

comparison_data = {
    'Model': ['Logistic Regression', 'Random Forest'],
    'Train Accuracy': [lr_train_acc, rf_train_acc],
    'Test Accuracy': [lr_test_acc, rf_test_acc]
}
comparison_df = pd.DataFrame(comparison_data)
print(f"\n{comparison_df.to_string(index=False)}")

# Visualize model comparison and feature importance
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle('Model Performance & Feature Importance', fontsize=14)

# Model comparison
models = comparison_df['Model']
train_acc = comparison_df['Train Accuracy']
test_acc = comparison_df['Test Accuracy']

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, train_acc, width, label='Train Accuracy', color='skyblue')
axes[0].bar(x + width/2, test_acc, width, label='Test Accuracy', color='lightcoral')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].set_ylim([0, 1])

# Feature importance
top_features = feature_importance.head(10)
axes[1].barh(top_features['Feature'], top_features['Importance'], color='lightgreen')
axes[1].set_title('Top 10 Feature Importance (Random Forest)')
axes[1].set_xlabel('Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=100, bbox_inches='tight')
print("\n[+] Model comparison plot saved as 'model_comparison.png'")
plt.close()

# ============= CONFUSION MATRIX =============
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Confusion Matrices', fontsize=14)

# Logistic Regression CM
lr_cm = confusion_matrix(y_test, lr_test_pred)
sns.heatmap(lr_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Logistic Regression')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Random Forest CM
rf_cm = confusion_matrix(y_test, rf_test_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1], cbar=False)
axes[1].set_title('Random Forest')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=100, bbox_inches='tight')
print("[+] Confusion matrices plot saved as 'confusion_matrices.png'")
plt.close()

# ============= SUMMARY =============
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"\n[+] Dataset loaded and processed: {len(df)} records")
print(f"[+] {len(feature_columns)} features engineered")
print(f"[+] 2 models trained and evaluated")
print(f"[+] Best Model: {'Random Forest' if rf_test_acc > lr_test_acc else 'Logistic Regression'} (Test Accuracy: {max(rf_test_acc, lr_test_acc):.4f})")
print(f"[+] Visualizations saved: f1_analysis.png, model_comparison.png, confusion_matrices.png")
print("\n" + "="*50)

# ============= PREDICTION FUNCTION =============
print("\n===== WINNER & PODIUM PREDICTION =====")

def predict_driver_performance(driver_name, grid_pos, quali_pos, last5_avg, career_races, career_wins, win_rate, grid_improvement, constructor_avg, season_points_per_race):
    """
    Predict if a driver will finish in top 3 (podium)
    
    Parameters:
    - driver_name: Driver name
    - grid_pos: Grid position (1-20)
    - quali_pos: Qualifying position (1-20)
    - last5_avg: Last 5 races average finish position
    - career_races: Career races count
    - career_wins: Career wins
    - win_rate: Win rate (0-1)
    - grid_improvement: Grid position - Finish position
    - constructor_avg: Team average finish position for season
    - season_points_per_race: Points per race this season
    
    Returns:
    - Prediction (Top 3 or Not Top 3)
    - Probability of Top 3 finish
    """
    
    features = np.array([[
        grid_pos, quali_pos, last5_avg, career_races, career_wins,
        win_rate, grid_improvement, constructor_avg, season_points_per_race
    ]])
    
    features_scaled = scaler.transform(features)
    
    # Get prediction from Logistic Regression (best model)
    prediction = lr_model.predict(features_scaled)[0]
    probability = lr_model.predict_proba(features_scaled)[0]
    
    top3_prob = probability[1] * 100
    
    result = "[TOP 3 FINISH]" if prediction == 1 else "[NOT TOP 3]"
    
    return {
        'Driver': driver_name,
        'Prediction': result,
        'Top 3 Probability': f"{top3_prob:.2f}%",
        'Win Likelihood (approx)': f"{top3_prob * 0.3:.2f}%" # Rough estimate
    }

# ============= EXAMPLE PREDICTIONS =============
print("\n===== EXAMPLE PREDICTIONS: =====")

# Get latest season drivers
latest_year = df['Year'].max()
latest_drivers = df[df['Year'] == latest_year].drop_duplicates('DriverName')[['DriverName', 'GridPosition', 'QualifyingPosition', 'Last5FinishAvg', 'CareerRaces', 'CareerWin', 'WinRate', 'GridImprovement', 'ConstructorAvgPosition', 'CurrentSeasonPointsPerRace']].head(5)

predictions_list = []

for idx, row in latest_drivers.iterrows():
    pred = predict_driver_performance(
        driver_name=row['DriverName'],
        grid_pos=row['GridPosition'],
        quali_pos=row['QualifyingPosition'],
        last5_avg=row['Last5FinishAvg'],
        career_races=row['CareerRaces'],
        career_wins=row['CareerWin'],
        win_rate=row['WinRate'],
        grid_improvement=row['GridImprovement'],
        constructor_avg=row['ConstructorAvgPosition'],
        season_points_per_race=row['CurrentSeasonPointsPerRace']
    )
    predictions_list.append(pred)
    print(f"Driver: {pred['Driver']}")
    print(f"  {pred['Prediction']}")
    print(f"  Top 3 Probability: {pred['Top 3 Probability']}")
    print()

# ============= TOP CONTENDERS =============
print("\n===== TOP PODIUM CONTENDERS (Latest Season): =====")

# Calculate prediction probability for all drivers in latest season
latest_df = df[df['Year'] == latest_year].copy()

predictions = []
for idx, row in latest_df.iterrows():
    features = np.array([[
        row['GridPosition'], row['QualifyingPosition'], row['Last5FinishAvg'],
        row['CareerRaces'], row['CareerWin'], row['WinRate'], row['GridImprovement'],
        row['ConstructorAvgPosition'], row['CurrentSeasonPointsPerRace']
    ]])
    
    features_scaled = scaler.transform(features)
    prob = lr_model.predict_proba(features_scaled)[0][1]
    
    predictions.append({
        'Driver': row['DriverName'],
        'Constructor': row['Constructor'],
        'Grid': int(row['GridPosition']),
        'Quali': int(row['QualifyingPosition']),
        'Top3_Probability': prob
    })

predictions_df = pd.DataFrame(predictions).drop_duplicates('Driver').sort_values('Top3_Probability', ascending=False)

print(f"{'Rank':<6} {'Driver':<25} {'Constructor':<15} {'Grid':<6} {'Quali':<6} {'Top 3 Prob':<15}")
print("-"*75)

for i, (idx, row) in enumerate(predictions_df.head(10).iterrows(), 1):
    print(f"{i:<6} {row['Driver']:<25} {row['Constructor']:<15} {row['Grid']:<6} {row['Quali']:<6} {row['Top3_Probability']*100:>6.2f}%")

print("\n" + "="*50)

# ============= 2025 SEASON PREDICTIONS =============
print("\n===== 2025 SEASON PREDICTIONS (UPCOMING) =====\n")

print("UPDATED WITH ACTUAL 2024 SEASON STATS\n")

# Get actual 2024 driver stats from the dataset
latest_year = 2024
latest_season_data = df[df['Year'] == latest_year].copy()

# Get unique drivers and their 2024 stats
drivers_2025_stats = []

for driver_name in latest_season_data['DriverName'].unique():
    driver_data = latest_season_data[latest_season_data['DriverName'] == driver_name]
    
    if len(driver_data) > 0:
        # Calculate average stats for 2024
        avg_grid = driver_data['GridPosition'].mean()
        avg_quali = driver_data['QualifyingPosition'].mean()
        last5_avg = driver_data['Last5FinishAvg'].iloc[-1] if len(driver_data) > 0 else driver_data['FinishPosition'].mean()
        carrer_races = driver_data['CareerRaces'].iloc[-1]
        career_wins = driver_data['CareerWin'].iloc[-1]
        win_rate = driver_data['WinRate'].iloc[-1]
        grid_imp = driver_data['GridImprovement'].mean()
        constructor_avg = driver_data['ConstructorAvgPosition'].mean()
        season_pts = driver_data['CurrentSeasonPointsPerRace'].iloc[-1]
        
        drivers_2025_stats.append({
            'Driver': driver_name,
            'GridPosition': avg_grid,
            'QualifyingPosition': avg_quali,
            'Last5FinishAvg': last5_avg,
            'CareerRaces': carrer_races,
            'CareerWin': career_wins,
            'WinRate': win_rate,
            'GridImprovement': grid_imp,
            'ConstructorAvgPosition': constructor_avg,
            'CurrentSeasonPointsPerRace': season_pts
        })

predictions_2025 = []

for driver_stats in drivers_2025_stats:
    features = np.array([[
        driver_stats['GridPosition'], 
        driver_stats['QualifyingPosition'], 
        driver_stats['Last5FinishAvg'],
        driver_stats['CareerRaces'], 
        driver_stats['CareerWin'], 
        driver_stats['WinRate'],
        driver_stats['GridImprovement'], 
        driver_stats['ConstructorAvgPosition'],
        driver_stats['CurrentSeasonPointsPerRace']
    ]])
    
    features_scaled = scaler.transform(features)
    prob = lr_model.predict_proba(features_scaled)[0][1]
    
    predictions_2025.append({
        'Driver': driver_stats['Driver'],
        'Grid': f"{driver_stats['GridPosition']:.1f}",
        'Quali': f"{driver_stats['QualifyingPosition']:.1f}",
        'Last5Avg': f"{driver_stats['Last5FinishAvg']:.2f}",
        'WinRate': f"{driver_stats['WinRate']:.3f}",
        'Top3_Probability': prob * 100
    })

predictions_2025_df = pd.DataFrame(predictions_2025).sort_values('Top3_Probability', ascending=False)

print(f"{'Rank':<6} {'Driver':<25} {'Grid':<8} {'Quali':<8} {'Last5':<8} {'Top 3 Prob':<15}")
print("-"*70)

for i, (idx, row) in enumerate(predictions_2025_df.head(15).iterrows(), 1):
    print(f"{i:<6} {row['Driver']:<25} {row['Grid']:<8} {row['Quali']:<8} {row['Last5Avg']:<8} {row['Top3_Probability']:>6.2f}%")

print("\n" + "="*50)
print("Note: 2025 predictions updated with actual 2024 dataset.")
print("="*50)

