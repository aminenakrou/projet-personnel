import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df):
    numeric_features = ['feature1','feature2','feature3']
    numeric_transformer = StandardScaler()
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)])
    X = df.drop('churn', axis=1)
    y = df['churn']
    X_processed = preprocessor.fit_transform(X)
    return X_processed, y
