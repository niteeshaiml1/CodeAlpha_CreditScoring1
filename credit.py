import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# Set a random seed for reproducibility
np.random.seed(42)

# --- Data Generation ---
data_size = 1500
data = {
    'income': np.random.normal(55000, 25000, data_size),
    'loan_amount': np.random.normal(18000, 7000, data_size),
    'credit_score': np.random.randint(550, 850, data_size),
    'age': np.random.randint(25, 65, data_size),
    'payment_history_missed': np.random.randint(0, 5, data_size)
}
df = pd.DataFrame(data)

# --- Feature Engineering ---
# A crucial feature for credit risk is the debt-to-income ratio
df['debt_to_income_ratio'] = df['loan_amount'] / df['income']

# Define the target variable: 'creditworthiness'
# A high debt-to-income ratio or low credit score indicates higher risk (1)
df['creditworthiness'] = ((df['debt_to_income_ratio'] > 0.45) | (df['credit_score'] < 650)).astype(int)

# Clean up infinite values that might result from income=0
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)

# Select the features and target for training
features = ['income', 'debt_to_income_ratio', 'credit_score', 'age', 'payment_history_missed']
target = 'creditworthiness'
X = df[features]
y = df[target]

# Split the data and train the model
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = LogisticRegression(random_state=42, solver='liblinear')
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'credit_scoring_model.pkl')
print("Model trained and saved as 'credit_scoring_model.pkl'")

# Load the trained model from the file
try:
    model = joblib.load('credit_scoring_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: The model file 'credit_scoring_model.pkl' was not found.")
    print("Please run the 'Model Training and Saving' script first to train and save the model.")
    exit()

# Get user input for a new applicant
print("\nEnter applicant details:")
try:
    income = float(input("Enter annual income: "))
    loan_amount = float(input("Enter requested loan amount: "))
    credit_score = int(input("Enter credit score: "))
    age = int(input("Enter age: "))
    payment_history_missed = int(input("Enter number of missed payments: "))

    # Calculate the derived feature
    debt_to_income_ratio = loan_amount / income

    # Create a DataFrame for the new applicant with the correct features and order
    new_applicant_data = {
        'income': [income],
        'debt_to_income_ratio': [debt_to_income_ratio],
        'credit_score': [credit_score],
        'age': [age],
        'payment_history_missed': [payment_history_missed]
    }

    # Ensure the feature columns are in the correct order for the model
    X_new = pd.DataFrame(new_applicant_data)

    # Make a prediction
    prediction = model.predict(X_new)
    probabilities = model.predict_proba(X_new)

    # Display the results
    print("\n--- Prediction Result ---")

    predicted_class = "High Risk" if prediction[0] == 1 else "Low Risk"
    high_risk_prob = probabilities[0][1] * 100
    low_risk_prob = probabilities[0][0] * 100

    print(f"Predicted Creditworthiness: {predicted_class}")
    print(f"Probability of being High Risk: {high_risk_prob:.2f}%")
    print(f"Probability of being Low Risk: {low_risk_prob:.2f}%")

except ValueError:
    print("Invalid input. Please enter numeric values for all fields.")
except ZeroDivisionError:
    print("Invalid input. Annual income cannot be zero.")