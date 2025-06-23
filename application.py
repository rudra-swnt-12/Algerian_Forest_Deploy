from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application 

# import ridge regressor & standard scaler pickle files
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))


@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/streamlit')
def streamlit_app():
    return render_template('streamlit.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoints():
    if request.method == 'POST':
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classses = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled = standard_scaler.transform([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classses, Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template('home.html')
    else:
        return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)