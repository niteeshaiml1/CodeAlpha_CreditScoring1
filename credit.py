import pandas as pdimport pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

# --- Important: This is the file that serves your model! ---

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow your HTML file to communicate with the server
CORS(app)

# Load the trained model from the file once, when the server starts
try:
    model = joblib.load('credit_scoring_model.pkl')
    print("Model loaded successfully. The server is ready.")
except FileNotFoundError:
    print("Error: The model file 'credit_scoring_model.pkl' was not found.")
    print("Please run your original Python script first to train and save the model.")
    exit()

@app.route('/predict', methods=['POST'])
def predict():
    """
    This function is the API endpoint that receives data from the frontend,
    makes a prediction, and returns the result as JSON.
    """
    try:
        # Get the JSON data sent from the HTML form
        data = request.get_json(force=True)
        
        # Extract features from the received data
        income = float(data['income'])
        loan_amount = float(data['loan_amount'])
        credit_score = int(data['credit_score'])
        age = int(data['age'])
        payment_history_missed = int(data['payment_history_missed'])

        # Calculate the derived feature, just like in your training script
        if income == 0:
            return jsonify({'error': 'Annual income cannot be zero.'}), 400
        debt_to_income_ratio = loan_amount / income

        # Create a DataFrame for the new applicant with the correct features and order
        X_new = pd.DataFrame({
            'income': [income],
            'debt_to_income_ratio': [debt_to_income_ratio],
            'credit_score': [credit_score],
            'age': [age],
            'payment_history_missed': [payment_history_missed]
        })

        # Make a prediction using the loaded model
        prediction = model.predict(X_new)
        probabilities = model.predict_proba(X_new)

        # Prepare the response data
        predicted_class = "High Risk" if prediction[0] == 1 else "Low Risk"
        high_risk_prob = float(probabilities[0][1])
        low_risk_prob = float(probabilities[0][0])
        
        # Return the results as a JSON object
        return jsonify({
            'predicted_class': predicted_class,
            'high_risk_prob': high_risk_prob,
            'low_risk_prob': low_risk_prob
        })

    except ValueError:
        return jsonify({'error': 'Invalid input. Please enter numeric values.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Run the server on localhost port 5000
    app.run(port=5000)


    print("Invalid input. Annual income cannot be zero.")
