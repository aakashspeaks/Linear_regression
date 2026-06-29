import pickle
from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open('linear_regression_model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/predict_api",methods=['POST'])
def predict():
    data = request.json['data']
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return jsonify({'prediction': prediction.tolist()})


if __name__ == "__main__":
    app.run(debug=True)