from flask import Flask, render_template, request
import pickle
import numpy as np

# Load the model
try:
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_flower():
    try:
        # Retrieve form data
        sepal_length = float(request.form.get('sepal_length'))
        sepal_width = float(request.form.get('sepal_width'))
        petal_length = float(request.form.get('petal_length'))
        petal_width = float(request.form.get('petal_width'))

        # Create feature array for prediction
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

        # Make prediction
        result = model.predict(features)[0]

        # Determine result label based on prediction
        if result == 1:
            result_label = "Iris-setosa"
        elif result == 2:
            result_label = "Iris-versicolor"
        elif result == 3:
            result_label = "Iris-virginica"
        else:
            result_label = "Unknown"

    except Exception as e:
        return render_template('error.html', error_message=f"Error processing request: {e}")

    return render_template('index.html', result=result_label)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
