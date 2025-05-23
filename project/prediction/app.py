from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the models and scalers for both Bitcoin and Ethereum
bitcoin_model = joblib.load('models/bitcoin/linear_regression_model.pkl')
ethereum_model = joblib.load('models/ethereum/gradient_boosting_model.pkl')
binance_model = joblib.load('models/binance/linear_regression_model.pkl')
doge_model = joblib.load('models/dogecoin/linear_regression_model.pkl')
cosmos_model = joblib.load('models/cosmos/linear_regression_model.pkl')
litecoin_model = joblib.load('models/litecoin/linear_regression_model.pkl')
stellar_model = joblib.load('models/stellar/linear_regression_model.pkl')
ripple_model = joblib.load('models/ripple/linear_regression_model.pkl')
cardano_model = joblib.load('models/cardano/linear_regression_model.pkl')
solana_model = joblib.load('models/solana/linear_regression_model.pkl')

bitcoin_scaler = joblib.load('scalers/bitcoin/scaler.pkl')
ethereum_scaler = joblib.load('scalers/ethereum/scaler.pkl')
binance_scaler = joblib.load('scalers/binance/scaler.pkl')
litecoin_scaler = joblib.load('scalers/litecoin/scaler.pkl')
solana_scaler = joblib.load('scalers/solana/scaler.pkl')

# Initialize last_100 for both Bitcoin and Ethereum
bitcoin_last_100 = np.load('last_100/bitcoin/last_100.npy')
ethereum_last_100 = np.load('last_100/ethereum/last_100.npy')
binance_last_100 = np.load('last_100/binance/last_100.npy')
doge_last_100 = np.load('last_100/dogecoin/last_100.npy')
cosmos_last_100 = np.load('last_100/cosmos/last_100.npy')
litecoin_last_100 = np.load('last_100/litecoin/last_100.npy')
stellar_last_100 = np.load('last_100/stellar/last_100.npy')
ripple_last_100 = np.load('last_100/ripple/last_100.npy')
cardano_last_100 = np.load('last_100/cardano/last_100.npy')
solana_last_100 = np.load('last_100/solana/last_100.npy')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global bitcoin_last_100, ethereum_last_100, binance_last_100, doge_last_100, cosmos_last_100, litecoin_last_100, stellar_last_100, ripple_last_100, cardano_last_100, solana_last_100
    
    # Get the input data from the form
    coin = request.form['coin']  # Get selected coin 
    num_days = int(request.form['num_days'])  # Number of days for prediction

    # Choose the appropriate model, scaler, and last_100 data based on selected coin
    if coin == 'Bitcoin':
        model = bitcoin_model
        scaler = bitcoin_scaler
        last_100 = bitcoin_last_100
        use_scaler = True
    elif coin == 'Ethereum':
        model = ethereum_model
        scaler = ethereum_scaler
        last_100 = ethereum_last_100
        use_scaler = True
    elif coin == 'Binance':
        model = binance_model
        scaler = binance_scaler
        last_100 = binance_last_100
        use_scaler = True
    elif coin == 'Dogecoin':
        model = doge_model
        scaler = None  # No scaler for Dogecoin
        last_100 = doge_last_100
        use_scaler = False
    elif coin == 'Cosmos':
        model = cosmos_model
        scaler = None  # No scaler for Cosmos
        last_100 = cosmos_last_100
        use_scaler = False
    elif coin == 'Litecoin':
        model = litecoin_model
        scaler = litecoin_scaler
        last_100 = litecoin_last_100
        use_scaler = True
    elif coin == 'Stellar':
        model = stellar_model
        scaler = None  # No scaler for Stellar
        last_100 = stellar_last_100
        use_scaler = False
    elif coin == 'Ripple':
        model = ripple_model
        scaler = None  # No scaler for Stellar
        last_100 = ripple_last_100
        use_scaler = False
    elif coin == 'Cardano':
        model = cardano_model
        scaler = None  # No scaler for Cardano
        last_100 = cardano_last_100
        use_scaler = False
    elif coin == 'Solana':
        model = solana_model
        scaler = solana_scaler
        last_100 = solana_last_100
        use_scaler = True

    # Generate future predictions
    future_predictions = []
    current_date = datetime.now()
    
    for _ in range(num_days):
        next_day = model.predict(last_100)
        if use_scaler:
            prediction = scaler.inverse_transform(next_day.reshape(1, -1))[0, 0]
        else:
            prediction = next_day[0, 0]  # Raw value for Dogecoin

        future_predictions.append({
            "date": (current_date + timedelta(days=len(future_predictions))).strftime("%Y-%m-%d"),
            "prediction": prediction
        })
        # Update last_100 after prediction
        last_100 = np.append(last_100[:, 1:], next_day.reshape(1, 1), axis=1)

    return render_template('index.html', predictions=future_predictions)

if __name__ == '__main__':
    app.run(debug=True)
