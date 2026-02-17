import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load model and scaler
model = pickle.load(open("Rainfall.pkl", "rb"))
scale = pickle.load(open("scale.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    try:
        MinTemp = float(request.form['MinTemp'])
        MaxTemp = float(request.form['MaxTemp'])
        Humidity = float(request.form['Humidity'])
        Pressure = float(request.form['Pressure'])
        WindSpeed = float(request.form['WindSpeed'])

        # create same structure as training dataset
        input_data = [[
            MinTemp,
            MaxTemp,
            0,                 # Rainfall (not in form)
            WindSpeed,
            Humidity,
            Humidity,
            Pressure,
            Pressure,
            (MinTemp+MaxTemp)/2,
            (MinTemp+MaxTemp)/2,
            0                  # RainToday
        ]]

        data = pd.DataFrame(input_data, columns=[
            'MinTemp','MaxTemp','Rainfall','WindGustSpeed',
            'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
            'Temp9am','Temp3pm','RainToday'
        ])

        data = scale.transform(data)
        prediction = model.predict(data)[0]

        if prediction == 1:
            return render_template("chance.html")
        else:
            return render_template("nochance.html")

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(debug=True)
