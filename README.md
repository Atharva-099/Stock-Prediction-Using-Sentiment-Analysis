This repository contains experiments with stock price forecasting using **ARIMA** and **ARMA** time series models.

---

## Project Structure
arima.ipynb # Notebook that uses ARIMA models (with differencing)
arma.ipynb # Notebook that uses ARMA models (without differencing)
arima.py # Helper functions for ARIMA (stationarity check, walk-forward validation, etc.)
sentiment_utils.py for Finbert sentiment analysis

requirements.txt # Python dependencies

python -m venv .venv
.venv\Scripts\activate   # On Windows
source .venv/bin/activate # On macOS/Linux

pip install -r requirements.txt


How It Works
#ARIMA (arima.ipynb):
Uses ARIMA (p, d, q) models.

Data is differenced (d > 0) to enforce stationarity.

Fits one model on the training data and forecasts into the future.

Limitation: Can appear as a flat line if over-differenced or not updated over time.

#ARMA (arma.ipynb):

Uses ARMA (p, q) models (d = 0).

Works directly on stationary data (like stock returns).

Often combined with walk-forward validation:

Retrains the model step by step as new data becomes available.

Predictions adapt better to recent market changes.

Produces more accurate-looking forecasts for short-term horizons.

Displays the dataset prediction two times, once using sentiment analysis and one without sentiment analysis.

Key Difference

arima.ipynb = static model, may flatten.

arma.ipynb = adaptive walk-forward, tracks actuals more closely.

## Notes

ARIMA is sensitive to differencing order (d). If d is too high, predictions flatten.

ARMA often works better on stock returns rather than raw prices.

