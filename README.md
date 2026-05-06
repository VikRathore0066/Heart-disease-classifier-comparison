# Heart Disease Prediction — Comparative ML Classifier Analysis

## Abstract
This study compares five classical machine learning classifiers for binary
heart disease prediction using the UCI Heart Disease dataset. The pipeline
covers exploratory data analysis, missing value imputation, feature encoding,
statistical feature selection, and model evaluation across multiple metrics.


## Dataset
- **Source:** [Heart Failure Prediction Dataset — Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Samples:** 918 | **Features:** 11 | **Target:** HeartDisease (0 = No, 1 = Yes)
- **File:** `data/heart.csv`


## Project Pipeline

### 1. Exploratory Data Analysis
- Distribution plots for Age, RestingBP, Cholesterol, MaxHR
- Class balance check for target variable
- Countplots for categorical features (Sex, ChestPainType, FastingBS)
- Boxplot and violin plot for Cholesterol and Age vs HeartDisease
- Correlation heatmap (numeric features)

### 2. Data Cleaning
- Detected and replaced zero values in `Cholesterol` and `RestingBP`
  with column mean (zeros are physiologically invalid)
- Verified no null values post-imputation

### 3. Preprocessing
- One-hot encoding of categorical features (`Sex`, `ChestPainType`,
  `RestingECG`, `ExerciseAngina`, `ST_Slope`) with `drop_first=True`
- StandardScaler applied to numeric columns:
  `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`

### 4. Feature Engineering & Selection
- **Pearson Correlation** for continuous features vs target
- **Chi-Square Test** (α = 0.05) for categorical features vs target
  — features with p < 0.05 retained

### 5. Model Training & Evaluation
Five classifiers trained on an 80/20 stratified split:

| Model               | Accuracy | F1 Score | ROC-AUC |
|--------------------|----------|----------|---------|
| SVM                 | 0.8641   | 0.8804   | **0.9424** |
| Logistic Regression | 0.8750   | 0.8878   | 0.9332  |
| Random Forest       | 0.8804   | 0.8932   | 0.9309  |
| KNN                 | **0.8859** | **0.8986** | 0.9266 |
| Decision Tree       | 0.7554   | 0.7692   | 0.7579  |


## Key Findings
- **SVM achieved the highest ROC-AUC (0.9424)**, making it the most
  reliable model for distinguishing high-risk patients — the preferred
  metric in medical classification tasks.
- **KNN led in Accuracy and F1**, but ROC-AUC is prioritized over
  these for imbalanced clinical data.
- **Decision Tree significantly underperformed** (ROC-AUC: 0.7579),
  likely due to overfitting without pruning.
- Logistic Regression offers the best interpretability-to-performance
  ratio, making it viable for clinical explainability requirements.


## Tech Stack
- Python 3.14.2
- pandas, numpy, matplotlib, seaborn
- scikit-learn, scipy


## How to Run
```bash
pip install -r requirements.txt
# Open notebooks/heart_disease_analysis.ipynb in Jupyter or VS Code
```


## Future Work
- Hyperparameter tuning with GridSearchCV
- Cross-validation for robust evaluation
- SHAP values for feature interpretability
