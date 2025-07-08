import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allows cross-origin access for APIs

# Load trained model
model = joblib.load('model.pkl')

# List of feature names in the exact order expected by the model
FEATURES = ['age', 'sex', 'trestbps', 'chol', 'fbs', 'thalch', 'exang', 'oldpeak', 'ca',
            'cp_atypical angina', 'cp_non-anginal', 'cp_typical angina',
            'restecg_normal', 'restecg_st-t abnormality',
            'slope_flat', 'slope_upsloping',
            'thal_normal', 'thal_reversable defect']


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.is_json:
            # JSON API request
            data = request.get_json()
            features = data.get('features', [])
        else:
            # HTML form submission
            features = [float(request.form.get(feature)) for feature in FEATURES]

        # Convert to numpy array and validate
        features_array = np.array(features, dtype=float).reshape(1, -1)

        # Predict
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0][1]

        result = "HAS Heart Disease" if prediction == 1 else "DOES NOT have Heart Disease"

        # If from HTML form, render result page
        if not request.is_json:
            return render_template('result.html', prediction=result)

        # If from API, return JSON
        return jsonify({
            'prediction': result,
            'probability': round(float(probability), 4)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)
