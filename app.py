from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)


# Load the model, data structure, and the scaler
def load_objects():
    try:
        with open('best_logistic_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('preprocessed_data.pkl', 'rb') as f:
            df = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, df, scaler
    except FileNotFoundError as e:
        print(f"⚠️ Error: Missing file - {e}")
        return None, None, None


model, df, scaler = load_objects()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict.html')
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from request
        data = request.get_json()

        # 2. Setup columns in the exact order the model expects
        if df is not None:
            # We drop the target variable to get the feature list
            cols = df.drop(columns=['e_signed'], errors='ignore').columns
        else:
            return jsonify({'error': 'Model structure (preprocessed_data.pkl) not loaded'}), 500

        # 3. Create a template DataFrame with zeros
        input_df = pd.DataFrame(0, index=[0], columns=cols)

        # 4. Map Numeric Inputs from JSON
        # Using realistic defaults to prevent "zero-bias" in probability
        input_df['age'] = int(data.get('age', 30))
        input_df['income'] = float(data.get('income', 50000))
        input_df['years_employed'] = float(data.get('years_employed', 2))
        input_df['current_address_year'] = float(data.get('current_address_year', 1))
        input_df['amount_requested'] = float(data.get('amount_requested', 1000))
        input_df['ext_quality_score'] = float(data.get('ext_quality_score', 50))
        input_df['ext_quality_score_2'] = float(data.get('ext_quality_score_2', 50))
        input_df['inquiries_last_month'] = int(data.get('inquiries_last_month', 0))
        input_df['personal_account_months'] = int(data.get('personal_account_months', 12))
        input_df['home_owner'] = int(data.get('home_owner', 0))
        input_df['has_debt'] = int(data.get('has_debt', 0))

        # Risk Scores
        for i in ['', '_2', '_3', '_4', '_5']:
            col_name = f'risk_score{i}'
            if col_name in input_df.columns:
                input_df[col_name] = float(data.get(col_name, 50))

        # 5. Map One-Hot Encoded Pay Schedule
        pay_schedule = data.get('pay_schedule', 'bi-weekly')
        pay_col = f"pay_schedule_{pay_schedule}"
        if pay_col in input_df.columns:
            input_df[pay_col] = 1

        # 6. Apply Scaling (CRITICAL for Logistic Regression)
        # We transform the data so $50,000 income doesn't overwhelm the weights
        if scaler is not None:
            features_to_predict = scaler.transform(input_df)
        else:
            return jsonify({'error': 'Scaler not found. Probability will be inaccurate.'}), 500

        # 7. Make Prediction
        prediction = model.predict(features_to_predict)
        prob = model.predict_proba(features_to_predict)[0][1]

        return jsonify({
            'prediction': int(prediction[0]),
            'probability': round(float(prob), 4),
            'mock': False
        })

    except Exception as e:
        print(f"❌ Error during prediction: {str(e)}")
        return jsonify({
            'error': str(e),
            'prediction': 0,
            'probability': 0.5,
            'mock': True
        }), 500


if __name__ == '__main__':
    app.run(debug=True)