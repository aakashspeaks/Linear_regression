import pickle
import pandas as pd
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the trained model
with open("linear_regression_model.pkl", "rb") as f:
    model = pickle.load(f)

# Feature names
COLUMNS = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT"
]


@app.route("/")
def home():
    return render_template("home1.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Read values from HTML form
        data = []

        for col in COLUMNS:
            value = request.form.get(col)

            if value is None or value.strip() == "":
                return render_template(
                    "home.html",
                    prediction_text=f"Missing value for {col}"
                )

            data.append(float(value))

        print("Received Data:", data)

        # Create DataFrame
        df = pd.DataFrame([data], columns=COLUMNS)

        print(df)

        # Predict
        prediction = model.predict(df)

        return render_template(
            "home.html",
            prediction_text=f"Predicted House Price: {prediction[0]:.2f}"
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return render_template(
            "home.html",
            prediction_text=f"Error: {str(e)}"
        )


@app.route("/predict_api", methods=["POST"])
def predict_api():
    try:
        json_data = request.get_json()

        if json_data is None:
            return jsonify({"error": "No JSON received"}), 400

        data = json_data.get("data")

        if data is None:
            return jsonify({"error": "Key 'data' not found"}), 400

        # Create DataFrame from dictionary
        df = pd.DataFrame([data])

        # Ensure correct column order
        df = df[COLUMNS]

        prediction = model.predict(df)

        return jsonify({
            "prediction": float(prediction[0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)