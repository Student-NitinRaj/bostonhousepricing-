import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# ---- Load Model and Scaler ----
# Load trained model, scaler, and feature order
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_order = pickle.load(open("features.pkl","rb"))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']

    # Arrange input according to training feature order
    ordered_data = [data[feat] for feat in feature_order]

    # Scale and predict
    new_data = scaler.transform([ordered_data])
    output = regmodel.predict(new_data)

    return jsonify(float(output[0]))


if __name__ == "__main__":
    app.run(debug=True)
