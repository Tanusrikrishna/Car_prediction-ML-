from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model and label encoders
with open("car_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        company = label_encoders["company"].transform([request.form["company"]])[0]
        year = int(request.form["year"])
        kms_driven = float(request.form["kms_driven"])
        fuel_type = label_encoders["fuel_type"].transform([request.form["fuel_type"]])[0]

        features = np.array([[company, year, kms_driven, fuel_type]])
        predicted_price = model.predict(features)[0]

        return render_template("index.html", prediction=f"Estimated Car Price: ₹{predicted_price:.2f}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
