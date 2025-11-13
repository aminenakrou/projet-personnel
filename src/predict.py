import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Jeu de données exemple
# Churn: 0 ou 1
# Colonnes fictives
pd.DataFrame({
    "feature1": [0.4, 0.6, 0.7, 0.2, 0.9],
    "feature2": [1, 2, 4, 2, 1],
    "feature3": [3, 5, 2, 1, 6],
    "churn": [0, 1, 0, 1, 1]
}).to_csv("data/churn_sample.csv", index=False)
