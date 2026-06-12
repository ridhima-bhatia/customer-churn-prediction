# Customer Churn Prediction using ANN

Predicting whether a telecom customer will churn using an Artificial Neural Network built with TensorFlow/Keras.

## Dataset
- Telco Customer Churn dataset (7,032 customers, 20 features)
- Target: Binary classification (Churn: Yes/No)

## Methodology
1. **Data Cleaning** - Handled missing values in TotalCharges column (11 blank entries converted and dropped)
2. **EDA** - Analyzed churn distribution, monthly charges patterns across churned vs retained customers
3. **Feature Engineering** - Label encoding, One-Hot Encoding of categorical variables (21 → 31 features)
4. **Scaling** - StandardScaler applied to normalize feature ranges before feeding to ANN
5. **Model Training** - ANN trained for 30 epochs with batch size 32, 80/20 train-test split
6. **Evaluation** - Accuracy, Precision, Recall, F1-Score, Confusion Matrix

## Key Findings
- Churned customers have ~20% higher average monthly charges
- Top 3 predictors: TotalCharges, tenure, MonthlyCharges
- Dataset is imbalanced (73% No Churn, 27% Churn)

## Model Architecture
- Input layer → Dense(64, ReLU) → Dropout(0.3) → Dense(32, ReLU) → Dropout(0.3) → Dense(1, Sigmoid)
- Optimizer: Adam | Loss: Binary Crossentropy

## Results
- Test Accuracy: **79%**
- Precision (No Churn): 0.83 | Recall (No Churn): 0.89
- Precision (Churn): 0.63 | Recall (Churn): 0.51

## Visualizations
![EDA](eda_plot%20.png)
![Training Curve](training_curve%20.png)
![Confusion Matrix](confusion_matrix%20.png)
![Feature Importance](feature_importance%20.png)

## Tech Stack
Python, TensorFlow, Keras, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn
