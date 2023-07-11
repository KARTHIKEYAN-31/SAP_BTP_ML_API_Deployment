import time
import pandas as pd
import numpy as np
import pycaret
from pycaret.classification import *
from flask import Flask, request, jsonify
import traceback


app = Flask(__name__)

port = 8080

@app.route('/predict', methods=['POST'])
def predict():

    qda = load_model("water_qda_model") # Load "model.pkl"

    data = request.get_json()

    ph = float(data['PH'])
    hardness = float(data['Hardness'])
    Solids = float(data['Solids'])
    Chloramines = float(data['Chloramines'])
    Sulfate = float(data['Sulfate'])
    Conductivity = float(data['Conductivity'])
    Organic_carbon = float(data['Organic_Carbon'])
    Trihalomethanes = float(data['Trihalomethanes'])
    Turbidity = float(data['Turbidity'])
    # Do something with the inputs

    dict_test = {'ph':[ph],'Hardness':[hardness],'Solids':[Solids],'Chloramines':[Chloramines],'Sulfate':[Sulfate],'Conductivity':[Conductivity],'Organic_carbon':[Organic_carbon],'Trihalomethanes':[Trihalomethanes],'Turbidity':[Turbidity]}

    test_df = pd.DataFrame.from_dict(dict_test)


    pred = predict_model(qda,data=test_df)
    prediction = list(pred['prediction_label'])[0]
    confidence = list(pred['prediction_score'])[0]

    return jsonify({'prediction': str(prediction),'Confidence':str(confidence)})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)