"""Flask app that serves the churn model: a web form plus a JSON API."""
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from train import FEATURES

app = Flask(__name__)
model = joblib.load("model.joblib")

PAGE = """
<!doctype html>
<title>Churn Predictor</title>
<h2>Customer Churn Predictor</h2>
<form method="post">
  {% for f in features %}
    <p><label>{{ f }}: <input name="{{ f }}" type="number" step="any" required></label></p>
  {% endfor %}
  <button>Predict</button>
</form>
{% if result %}<h3>Churn risk: {{ result }}%</h3>{% endif %}
"""


def predict(values: dict) -> float:
    row = pd.DataFrame([{f: float(values[f]) for f in FEATURES}])
    return round(float(model.predict_proba(row)[0][1]) * 100, 1)


@app.route("/", methods=["GET", "POST"])
def home():
    result = predict(request.form) if request.method == "POST" else None
    return render_template_string(PAGE, features=FEATURES, result=result)


@app.post("/api/predict")
def api_predict():
    data = request.get_json(silent=True) or {}
    missing = [f for f in FEATURES if f not in data]
    if missing:
        return jsonify(error=f"missing fields: {missing}"), 400
    return jsonify(churn_probability_percent=predict(data))


if __name__ == "__main__":
    app.run(debug=True)
