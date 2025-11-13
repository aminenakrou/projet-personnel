from flask import Flask, request, jsonify
import pandas as pd
from model_utils import load_model, predict
import os

app = Flask(__name__)
model_path = "stacking-model"

@app.route('/predict', methods=['POST'])
def predict_api():
    input_json = request.json
    input_df = pd.DataFrame([input_json])
    model = load_model(model_path)
    pred = predict(model, input_df)
    return jsonify({"churn_prediction": int(pred[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
