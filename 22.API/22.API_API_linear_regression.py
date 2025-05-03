from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)
# Define the path to the model file
model_folder = '../model'
model_path = os.path.join(model_folder, 'linear_regression_model.pkl')
# Load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)
m = model['slope']
b = model['intercept']


@app.route('/')
def home():
    return "Welcome to the linear regression prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)
    if 'x' not in data:
        return jsonify({"error": "Invalid input, 'x' value required"}), 400
    x_values = np.array(data['x'], dtype=np.float64)
    predictions = [(m * x) + b for x in x_values]
    return jsonify({"predictions": predictions})


if __name__ == '__main__':
    app.run(debug=True)




'''
PS D:\SHIVANI_PATEL\Python\AIML WIPRO\ML_Python> curl.exe -X POST -H "Content-Type: application/json" -d '{\"x\":[2, 4, 6]}' http://127.0.0.1:5000/predict
{
  "predictions": [
    24.0,
    44.0,
    64.0
  ]
}

'''