import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# ---- Load Model, Scaler, and Feature Order ----
regmodel = pickle.load(open("regmodel.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_order = pickle.load(open("features.pkl", "rb"))

# ---- Home Page ----
@app.route('/')
def home():
    return render_template('home.html')

# ---- API Prediction (Postman/JSON) ----
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']  # JSON input

    # Arrange input according to training feature order
    ordered_data = [data[feat] for feat in feature_order]

    # Scale and predict
    new_data = scaler.transform([ordered_data])
    output = regmodel.predict(new_data)

    return jsonify(float(output[0]))

# ---- Form Prediction (HTML UI) ----
@app.route('/predict', methods=['POST'])
def predict():
    # Collect values from HTML form
    data = [float(x) for x in request.form.values()]

    # Scale and predict
    final_input = scaler.transform(np.array(data).reshape(1, -1))
    output = regmodel.predict(final_input)[0]

    return render_template(
        "home.html",
        prediction_text=f"The House price prediction is {output:.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)








# import pickle
# from flask import Flask, request, app, jsonify, url_for, render_template
# import numpy as np
# import pandas as pd

# app = Flask(__name__)

# # ---- Load Model and Scaler ----
# # Load trained model, scaler, and feature order
# regmodel = pickle.load(open("regmodel.pkl", "rb"))
# scaler = pickle.load(open("scaler.pkl", "rb"))
# feature_order = pickle.load(open("features.pkl","rb"))

# @app.route('/predict_api', methods=['POST'])
# def predict_api():
#     data = request.json['data']

#     # Arrange input according to training feature order
#     ordered_data = [data[feat] for feat in feature_order]

#     # Scale and predict
#     new_data = scaler.transform([ordered_data])
#     output = regmodel.predict(new_data)

#     return jsonify(float(output[0]))
# @app.route('/predict', methods=['POST'] )
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input= scaler.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text= "The House price prediction is {}".format(output))

# if __name__ == "__main__":
#     app.run(debug=True)
