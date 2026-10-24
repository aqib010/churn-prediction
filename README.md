# Customer Churn Prediction

Trains a Random Forest to predict customer churn and serves it through a Flask web form and a JSON API.

## Run
    pip install -r requirements.txt
    python train.py        # creates data.csv and model.joblib, prints metrics
    python app.py          # open http://127.0.0.1:5000

API example:
    curl -X POST localhost:5000/api/predict -H "Content-Type: application/json" \
      -d '{"tenure_months":3,"monthly_charges":95,"support_calls":6,"contract_months":1}'

## Important
The data is synthetic (generated in train.py), so metrics only show the pipeline works. For a stronger project, swap in the Kaggle "Telco Customer Churn" dataset, do proper preprocessing and compare models.
