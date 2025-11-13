import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn

# Chargement des données
df = pd.read_csv('../data/churn_sample.csv')
X = df.drop('churn', axis=1)
y = df['churn']

# Split
test_size = 0.2
random_state = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Stacking
level0 = [
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier())
]
level1 = LogisticRegression()
model = StackingClassifier(estimators=level0, final_estimator=level1)

mlflow.set_tracking_uri("http://localhost:5001")
with mlflow.start_run():
    model.fit(X_train, y_train)
    mlflow.sklearn.log_model(model, "stacking-model")
    score = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", score)
    print("Accuracy:", score)
