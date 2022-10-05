import json
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
from app import app

app = Flask(__name__)

#Loading the pickle model
forest_model = pickle.load(open("forestmodel.pkl","rb"))
scaler_model = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/prediction_api", methods = ["POST"])
def prediction_api():
    data = request.json["data"]
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    newdata = scaler_model.transform(np.array(list(data.values())).reshape(1, -1))
    output = forest_model.predict(newdata)
    print(output[0])
    return jsonify(output[0])

@app.route("/predict_api", methods = ["POST"])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaler_model.transform(np.array(data).reshape(1, -1))
    print(final_input)
    output = forest_model.predict(final_input)[0]
    return render_template("home.html", prediction_text = "The Price Prediction of a house meeting the following conditions is {}".format(output))


if __name__=="__main__":
    app.run(host = "localhost", port=3000, debug=True)
