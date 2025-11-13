import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

# Utilitaire de chargement du modèle et prédiction
def load_model(path):
    import mlflow.sklearn
    return mlflow.sklearn.load_model(path)

def predict(model, input_df):
    return model.predict(input_df)
